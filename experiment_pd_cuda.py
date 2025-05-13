import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import threading
import pynvml  # 用于GPU监控

# --- 配置参数 ---
MODEL_NAME = "gpt2"  # 你可以换成更大的模型如 "gpt2-medium", "gpt2-large" 等，但需要更多显存
# MODEL_NAME = "Qwen/Qwen3-32B" # 如果你有足够显存和权限可以尝试
NUM_CONCURRENT_REQUESTS = 5  # 并发请求数 (批处理大小)
TARGET_PROMPT_TOKENS = 256  # 每个请求的目标输入Token数
MAX_NEW_TOKENS = 256  # 每个请求的最大生成Token数
GPU_MONITOR_INTERVAL = 0.1  # GPU监控采样间隔（秒）
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --- GPU监控线程 ---
class GPUMonitor:
    def __init__(self, interval_seconds=0.1):
        self.interval_seconds = interval_seconds
        self.stop_event = threading.Event()
        self.utilization_log = []
        self.memory_log = []
        self.thread = None
        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # 假设使用GPU 0
            self.monitoring_possible = True
            print("pynvml初始化成功，将监控GPU。")
        except Exception as e:
            self.monitoring_possible = False
            print(f"pynvml初始化失败，无法监控GPU: {e}")
            print("请确保已安装pynvml库 (pip install pynvml) 并且有NVIDIA驱动。")

    def _monitor(self):
        while not self.stop_event.is_set():
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                self.utilization_log.append(util.gpu)
                self.memory_log.append(mem_info.used / (1024 ** 2))  # MB
            except Exception as e:
                # print(f"GPU监控错误: {e}") # 避免过多打印
                self.utilization_log.append(-1)  # 标记错误
                self.memory_log.append(-1)
            time.sleep(self.interval_seconds)

    def start(self):
        if not self.monitoring_possible:
            return
        self.stop_event.clear()
        self.utilization_log = []
        self.memory_log = []
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()
        # print("GPU监控线程已启动。")

    def stop(self):
        if not self.monitoring_possible or self.thread is None:
            return
        self.stop_event.set()
        self.thread.join(timeout=self.interval_seconds * 5)  # 等待线程结束
        # print("GPU监控线程已停止。")
        if not self.utilization_log:  # 如果没有收集到数据
            return 0, 0

        valid_utils = [u for u in self.utilization_log if u != -1]
        avg_util = sum(valid_utils) / len(valid_utils) if valid_utils else 0

        valid_mems = [m for m in self.memory_log if m != -1]
        avg_mem = sum(valid_mems) / len(valid_mems) if valid_mems else 0
        return avg_util, avg_mem

    def __del__(self):
        if self.monitoring_possible:
            try:
                pynvml.nvmlShutdown()
                # print("pynvml已关闭。")
            except:
                pass


# --- 辅助函数：生成指定长度的Prompt ---
def generate_prompt_batch(tokenizer, target_token_count, num_prompts):
    """生成一批prompt，每个prompt大约有target_token_count个token"""
    # 使用一个简单的重复词汇来构造，实际应用中会是真实文本
    base_text = "这是一个用于测试长文本的示例句子，我们会重复它直到达到目标长度。"
    base_tokens = tokenizer.encode(base_text, add_special_tokens=False)

    # 计算需要重复多少次基本单元
    if not base_tokens:  # 如果基础文本是空的或者只包含特殊token
        base_text = "测试"
        base_tokens = tokenizer.encode(base_text, add_special_tokens=False)
        if not base_tokens:  # 极端情况
            base_tokens = [tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0]

    num_repeats = (target_token_count + len(base_tokens) - 1) // len(base_tokens)

    single_prompt_tokens = (base_tokens * num_repeats)[:target_token_count]
    single_prompt_text = tokenizer.decode(single_prompt_tokens)

    return [single_prompt_text] * num_prompts


# --- 主实验逻辑 ---
def run_experiment():
    if DEVICE == "cpu":
        print("警告：未检测到CUDA GPU，将在CPU上运行。结果可能不具代表性且非常缓慢。")

    print(
        f"实验配置: 模型={MODEL_NAME}, 并发数={NUM_CONCURRENT_REQUESTS}, 输入长度约={TARGET_PROMPT_TOKENS}, 生成长度={MAX_NEW_TOKENS}")

    # 1. 加载模型和分词器
    print("正在加载模型和分词器...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
        model.eval()  # 设置为评估模式
    except Exception as e:
        print(f"加载模型或分词器失败: {e}")
        print("请检查模型名称是否正确，网络是否连接，或是否有足够磁盘空间/权限。")
        return

    # 如果tokenizer没有pad_token，则设置为eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print(f"Tokenizer未设置pad_token，已自动设为eos_token_id: {tokenizer.eos_token_id}")

    # 2. 准备输入数据
    print(f"正在准备 {NUM_CONCURRENT_REQUESTS} 路并发输入数据...")
    prompts_batch = generate_prompt_batch(tokenizer, TARGET_PROMPT_TOKENS, NUM_CONCURRENT_REQUESTS)

    # 对整个批次进行分词
    # 注意：HuggingFace的padding默认是向右填充(padding_side='right')
    # 对于自回归模型，通常期望pad token在左边，但generate函数内部会处理attention_mask
    inputs = tokenizer(prompts_batch, return_tensors="pt", padding=True, truncation=True,
                       max_length=TARGET_PROMPT_TOKENS).to(DEVICE)
    input_ids_batch = inputs.input_ids
    attention_mask_batch = inputs.attention_mask
    actual_prompt_tokens_per_request = input_ids_batch.shape[1]  # 获取实际的输入token数（padding后）
    total_prompt_tokens = NUM_CONCURRENT_REQUESTS * actual_prompt_tokens_per_request
    print(
        f"输入数据准备完成。每个请求实际输入token数: {actual_prompt_tokens_per_request} (总输入: {total_prompt_tokens} tokens)")

    gpu_monitor = GPUMonitor(interval_seconds=GPU_MONITOR_INTERVAL)

    # 3. GPU预热 (非常重要，避免测量首次编译CUDA核的开销)
    print("正在进行GPU预热...")
    if DEVICE == "cuda":
        try:
            _ = model.generate(
                input_ids_batch[:, :min(actual_prompt_tokens_per_request, 10)],  # 使用部分输入进行预热
                attention_mask=attention_mask_batch[:, :min(actual_prompt_tokens_per_request, 10)],
                max_new_tokens=5,
                pad_token_id=tokenizer.pad_token_id
            )
            torch.cuda.synchronize()  # 确保GPU操作完成
            print("GPU预热完成。")
        except Exception as e:
            print(f"GPU预热期间发生错误: {e}")
            print("请检查模型和输入是否与GPU兼容，或显存是否足够。")
            return

    # --- 开始正式测试 ---
    print("\n--- 开始正式测试 ---")

    # 4. 测试 "Prefill + 首个Token Decode" 阶段 (近似Prefill阶段)
    print(f"\n阶段1: 测试近似Prefill性能 (生成1个新token)")
    gpu_monitor.start()
    time.sleep(0.1)  # 给监控线程一点时间启动

    t_start_prefill_phase = time.perf_counter()
    if DEVICE == "cuda": torch.cuda.synchronize()

    generated_ids_first_step = model.generate(
        input_ids_batch,
        attention_mask=attention_mask_batch,
        max_new_tokens=1,
        pad_token_id=tokenizer.pad_token_id,
        # 关闭采样和beam search以获得更稳定的性能
        do_sample=False,
        num_beams=1
    )

    if DEVICE == "cuda": torch.cuda.synchronize()
    t_end_prefill_phase = time.perf_counter()

    time.sleep(0.1)  # 等待最后的监控数据
    avg_util_prefill, avg_mem_prefill = gpu_monitor.stop()

    time_prefill_phase = t_end_prefill_phase - t_start_prefill_phase
    num_generated_in_prefill_step = (generated_ids_first_step.shape[
                                         1] - actual_prompt_tokens_per_request) * NUM_CONCURRENT_REQUESTS

    # Prefill阶段处理的是输入token，生成的是第一个新token
    # 我们主要关注输入token的处理速度作为Prefill的指标
    throughput_prefill_phase = total_prompt_tokens / time_prefill_phase

    print(f"  Prefill阶段（及首个token生成）耗时: {time_prefill_phase:.4f} 秒")
    print(f"  Prefill阶段输入Token总数: {total_prompt_tokens}")
    print(f"  Prefill阶段（近似）吞吐量: {throughput_prefill_phase:.2f} tokens/秒")
    if gpu_monitor.monitoring_possible:
        print(f"  Prefill阶段平均GPU利用率: {avg_util_prefill:.2f}%")
        print(f"  Prefill阶段平均GPU显存使用: {avg_mem_prefill:.2f} MB")

    # 5. 测试完整生成阶段 (MAX_NEW_TOKENS)
    print(f"\n阶段2: 测试完整生成性能 (生成 {MAX_NEW_TOKENS} 个新token)")
    # 清理显存，可选，但有时有帮助
    if DEVICE == "cuda":
        del generated_ids_first_step
        torch.cuda.empty_cache()
        time.sleep(0.5)  # 等待显存清理

    gpu_monitor.start()
    time.sleep(0.1)

    t_start_full_generation = time.perf_counter()
    if DEVICE == "cuda": torch.cuda.synchronize()

    generated_ids_full = model.generate(
        input_ids_batch,
        attention_mask=attention_mask_batch,
        max_new_tokens=MAX_NEW_TOKENS,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=False,
        num_beams=1
    )

    if DEVICE == "cuda": torch.cuda.synchronize()
    t_end_full_generation = time.perf_counter()

    time.sleep(0.1)
    avg_util_full_gen, avg_mem_full_gen = gpu_monitor.stop()

    time_full_generation = t_end_full_generation - t_start_full_generation

    # 实际生成的token数可能因遇到EOS等而少于MAX_NEW_TOKENS
    actual_new_tokens_per_request_full = generated_ids_full.shape[1] - actual_prompt_tokens_per_request
    total_new_tokens_generated_full = actual_new_tokens_per_request_full * NUM_CONCURRENT_REQUESTS

    print(f"  完整生成 ({actual_new_tokens_per_request_full}个新tokens/请求) 耗时: {time_full_generation:.4f} 秒")
    print(f"  完整生成阶段新生成Token总数: {total_new_tokens_generated_full}")
    if gpu_monitor.monitoring_possible:
        print(f"  完整生成阶段平均GPU利用率: {avg_util_full_gen:.2f}%")
        print(f"  完整生成阶段平均GPU显存使用: {avg_mem_full_gen:.2f} MB")

    # 6. 近似计算Decode阶段性能
    # Decode阶段时间 = 完整生成时间 - (Prefill + 首个Token Decode)时间
    # Decode阶段处理的token = 完整生成的新token数 - 首个Token Decode阶段生成的token数 (即1 * NUM_CONCURRENT_REQUESTS)

    # 如果完整生成实际产生的token数少于等于1，则无法计算后续Decode阶段
    if actual_new_tokens_per_request_full <= 1 and MAX_NEW_TOKENS > 1:
        print("\n  警告: 完整生成阶段实际生成的token数不足以计算后续Decode性能。")
        time_decode_phase_approx = float('nan')
        throughput_decode_phase_approx = float('nan')
        total_decode_phase_tokens = 0
    elif MAX_NEW_TOKENS == 1:  # 如果本身就只要求生成1个token，那么就没有“后续”的decode阶段了
        print("\n  提示: max_new_tokens=1, 没有后续的Decode阶段可供分析。")
        time_decode_phase_approx = float('nan')
        throughput_decode_phase_approx = float('nan')
        total_decode_phase_tokens = 0
    else:
        # 近似后续Decode阶段的时间
        # (完整生成时间 - "Prefill+1st token"时间)
        time_decode_phase_approx = time_full_generation - time_prefill_phase

        # 后续Decode阶段生成的token数
        # (总共生成的新token - 第一个decode step生成的token)
        # 第一个decode step为每个请求生成1个token，总共 NUM_CONCURRENT_REQUESTS 个
        total_decode_phase_tokens = total_new_tokens_generated_full - (1 * NUM_CONCURRENT_REQUESTS)

        if time_decode_phase_approx <= 0 or total_decode_phase_tokens <= 0:  # 避免除零或无效值
            print(
                f"\n  无法计算有效的后续Decode吞吐量 (时间: {time_decode_phase_approx:.4f}s, tokens: {total_decode_phase_tokens})")
            throughput_decode_phase_approx = float('nan')
        else:
            throughput_decode_phase_approx = total_decode_phase_tokens / time_decode_phase_approx
            print(f"\n  近似后续Decode阶段 (除去首个token后):")
            print(f"    耗时: {time_decode_phase_approx:.4f} 秒")
            print(f"    生成Token数: {total_decode_phase_tokens}")
            print(f"    吞吐量: {throughput_decode_phase_approx:.2f} tokens/秒")

    # --- 结果总结与分析 ---
    print("\n--- 实验结果总结与分析 ---")
    print(f"模型: {MODEL_NAME}, 并发数: {NUM_CONCURRENT_REQUESTS}")
    print(
        f"输入长度/请求 (目标): ~{TARGET_PROMPT_TOKENS} tokens, (实际padding后): {actual_prompt_tokens_per_request} tokens")
    print(f"生成长度/请求 (目标): {MAX_NEW_TOKENS} tokens, (实际): {actual_new_tokens_per_request_full} tokens")

    print("\n性能对比:")
    print(f"  1. 近似Prefill阶段 (处理输入 {total_prompt_tokens} tokens 并生成首批 {NUM_CONCURRENT_REQUESTS} tokens):")
    print(f"     - 耗时: {time_prefill_phase:.4f} s")
    print(f"     - 输入吞吐量: {throughput_prefill_phase:.2f} input tokens/s")
    if gpu_monitor.monitoring_possible: print(f"     - 平均GPU利用率: {avg_util_prefill:.2f}%")

    print(f"\n  2. 近似后续Decode阶段 (生成后续 {total_decode_phase_tokens} tokens):")
    print(f"     - 耗时: {time_decode_phase_approx:.4f} s")
    print(f"     - 输出吞吐量: {throughput_decode_phase_approx:.2f} output tokens/s")
    if gpu_monitor.monitoring_possible: print(f"     - (参考完整生成阶段的平均GPU利用率: {avg_util_full_gen:.2f}%)")

    print("\n分析:")
    if throughput_decode_phase_approx < throughput_prefill_phase:
        ratio = throughput_prefill_phase / throughput_decode_phase_approx if throughput_decode_phase_approx > 0 else float(
            'inf')
        print(f"  观察到Decode阶段的输出吞吐量 ({throughput_decode_phase_approx:.2f} tokens/s) "
              f"显著低于Prefill阶段的输入吞吐量 ({throughput_prefill_phase:.2f} tokens/s)。")
        print(f"  Prefill吞吐量约是Decode吞吐量的 {ratio:.2f} 倍。")
        print("  这表明，在Decode阶段，尽管GPU可能仍然保持一定的利用率（尤其是在高并发下），"
              "但每个token的处理速度变慢了。这主要是因为：")
        print("    a. 自回归特性：每个token的生成依赖于前一个，限制了单序列内的并行度。")
        print("    b. 内存带宽限制：频繁读写KV Cache可能成为瓶颈，尤其当上下文变长、并发请求多时。")
        print("    c. 计算密度较低：相比Prefill阶段大规模处理输入，Decode阶段每个step的计算量相对较小。")
        print(
            "  这种效率差异正是PD（Prefill-Decode）分离优化（如使用专门的kernel、PagedAttention、Continuous Batching等）所要解决的核心问题，")
        print("  旨在提高Decode阶段的效率，从而提升整体的LLM服务吞吐和响应速度。")
    elif throughput_decode_phase_approx is not float('nan'):
        print("  Decode阶段吞吐量与Prefill阶段吞吐量差异不明显或更高。这可能是由于：")
        print("    - 模型非常小，或者输入/输出序列非常短，导致性能差异不突出。")
        print("    - 特定的并发数和硬件组合导致了不同的瓶颈表现。")
        print("    - 近似测量方法引入的误差。")
        print("  请尝试增大模型、序列长度或调整并发数以观察更典型的现象。")
    else:
        print("  未能有效计算Decode阶段吞吐量，无法进行比较。")

    # 清理资源
    del model
    del tokenizer
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    print("\n实验结束，资源已清理。")


if __name__ == "__main__":
    if DEVICE == "cpu" and input("警告: 将在CPU上运行，可能非常慢。是否继续? (y/n): ").lower() != 'y':
        print("已取消实验。")
    else:
        run_experiment()