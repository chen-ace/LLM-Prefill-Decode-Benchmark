# 本实验对比Prefill和Decoding阶段的MPS性能，来说明Decoding在GPU上进行会极大浪费GPU的算力。
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

# 注意：移除了 threading 和 pynvml，因为pynvml不适用于MPS

# --- 配置参数 ---
MODEL_NAME = "gpt2"  # 你可以换成更大的模型如 "gpt2-medium", "gpt2-large" 等
NUM_CONCURRENT_REQUESTS = 5  # 并发请求数 (批处理大小)
TARGET_PROMPT_TOKENS = 256  # 每个请求的目标输入Token数
MAX_NEW_TOKENS = 256  # 每个请求的最大生成Token数
# DEVICE 在后面动态设置

# --- 设备检测 ---
if torch.backends.mps.is_available():
    DEVICE = "mps"
    print("检测到Apple Silicon MPS 后端可用，将使用 'mps' 设备。")
elif torch.cuda.is_available():  # 保留CUDA作为备选，以防脚本在非Mac的NVIDIA环境运行
    DEVICE = "cuda"
    print("检测到NVIDIA CUDA 后端可用，将使用 'cuda' 设备。")
    print("提示：如果您在Apple Silicon上，但此脚本检测到CUDA，可能是环境配置问题。")
else:
    DEVICE = "cpu"
    print("警告：未检测到MPS或CUDA GPU，将在CPU上运行。结果可能不具代表性且非常缓慢。")


# --- MPS GPU监控提示 ---
class MPSGPUMonitorInfo:
    def __init__(self):
        self.monitoring_possible = (DEVICE == "mps")  # 标记是否为MPS设备以显示提示
        if self.monitoring_possible:
            print("-" * 30)
            print("Apple Silicon (MPS) GPU活动监控提示:")
            print("此脚本不包含自动MPS GPU利用率跟踪。您可以通过以下方式手动观察：")
            print("1. 打开“活动监视器” -> 菜单栏“窗口” -> “GPU历史记录”。")
            print("2. 在终端运行: sudo powermetrics --samplers gpu_power -i 1000")
            print("   (按 Ctrl+C 停止 powermetrics)。")
            print("请在脚本的关键阶段（如模型生成时）观察GPU活动。")
            print("-" * 30)

    def start(self):
        # MPS版本无自动监控线程
        if self.monitoring_possible:
            print("提示：请手动开始观察MPS GPU活动。")
        pass

    def stop(self):
        # MPS版本无自动监控线程，返回占位符
        if self.monitoring_possible:
            print("提示：请手动结束观察MPS GPU活动。")
        return float('nan'), float('nan')  # 返回 NaN 表示无自动数据


# --- 辅助函数：生成指定长度的Prompt ---
def generate_prompt_batch(tokenizer, target_token_count, num_prompts):
    """生成一批prompt，每个prompt大约有target_token_count个token"""
    base_text = "这是一个用于测试长文本的示例句子，我们会重复它直到达到目标长度。"
    base_tokens = tokenizer.encode(base_text, add_special_tokens=False)

    if not base_tokens:
        base_text = "测试"
        base_tokens = tokenizer.encode(base_text, add_special_tokens=False)
        if not base_tokens:
            base_tokens = [tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0]

    num_repeats = (target_token_count + len(base_tokens) - 1) // len(base_tokens)
    single_prompt_tokens = (base_tokens * num_repeats)[:target_token_count]
    single_prompt_text = tokenizer.decode(single_prompt_tokens)

    return [single_prompt_text] * num_prompts


# --- 设备同步函数 ---
def synchronize_device(device_name):
    if device_name == "cuda":
        torch.cuda.synchronize()
    elif device_name == "mps":
        torch.mps.synchronize()
    # CPU 不需要显式同步


# --- 清理显存函数 (MPS上效果有限) ---
def empty_cache_device(device_name):
    if device_name == "cuda":
        torch.cuda.empty_cache()
    elif device_name == "mps":
        # torch.mps.empty_cache() # PyTorch 2.0+ MPS有此函数，但效果可能不如CUDA的
        # 对于旧版本或更通用的做法，依赖Python GC
        import gc
        gc.collect()
        pass  # MPS主要依赖变量删除和Python的垃圾回收


# --- 主实验逻辑 ---
def run_experiment():
    if DEVICE == "cpu" and not torch.backends.mps.is_available() and not torch.cuda.is_available():  # 再次确认
        print("警告：未检测到MPS或CUDA GPU，将在CPU上运行。结果可能不具代表性且非常缓慢。")

    print(
        f"实验配置: 模型={MODEL_NAME}, 并发数={NUM_CONCURRENT_REQUESTS}, 输入长度约={TARGET_PROMPT_TOKENS}, 生成长度={MAX_NEW_TOKENS}, 设备={DEVICE}")

    # 1. 加载模型和分词器
    print("正在加载模型和分词器...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        # 对于MPS，有时使用float16会有问题或不支持，默认float32
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
        model.eval()  # 设置为评估模式
    except Exception as e:
        print(f"加载模型或分词器失败: {e}")
        print("请检查模型名称是否正确，网络是否连接，或是否有足够磁盘空间/权限。")
        if DEVICE == "mps" and "does not support BFloat16" in str(e).lower():
            print("提示：部分模型在MPS上可能默认尝试使用BFloat16而出错，可尝试强制使用Float32。")
        return

    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print(f"Tokenizer未设置pad_token，已自动设为eos_token_id: {tokenizer.eos_token_id}")

    # 2. 准备输入数据
    print(f"正在准备 {NUM_CONCURRENT_REQUESTS} 路并发输入数据...")
    prompts_batch = generate_prompt_batch(tokenizer, TARGET_PROMPT_TOKENS, NUM_CONCURRENT_REQUESTS)
    inputs = tokenizer(prompts_batch, return_tensors="pt", padding=True, truncation=True,
                       max_length=TARGET_PROMPT_TOKENS).to(DEVICE)
    input_ids_batch = inputs.input_ids
    attention_mask_batch = inputs.attention_mask
    actual_prompt_tokens_per_request = input_ids_batch.shape[1]
    total_prompt_tokens = NUM_CONCURRENT_REQUESTS * actual_prompt_tokens_per_request
    print(
        f"输入数据准备完成。每个请求实际输入token数: {actual_prompt_tokens_per_request} (总输入: {total_prompt_tokens} tokens)")

    mps_monitor_info = MPSGPUMonitorInfo()  # 显示MPS监控提示

    # 3. GPU预热
    print("正在进行设备预热...")
    if DEVICE != "cpu":
        try:
            _ = model.generate(
                input_ids_batch[:, :min(actual_prompt_tokens_per_request, 10)],
                attention_mask=attention_mask_batch[:, :min(actual_prompt_tokens_per_request, 10)],
                max_new_tokens=5,
                pad_token_id=tokenizer.pad_token_id
            )
            synchronize_device(DEVICE)
            print("设备预热完成。")
        except Exception as e:
            print(f"设备预热期间发生错误: {e}")
            return

    # --- 开始正式测试 ---
    print("\n--- 开始正式测试 ---")
    mps_monitor_info.start()  # 提示用户开始手动观察

    # 4. 测试 "Prefill + 首个Token Decode" 阶段 (近似Prefill阶段)
    print(f"\n阶段1: 测试近似Prefill性能 (生成1个新token)")

    t_start_prefill_phase = time.perf_counter()
    synchronize_device(DEVICE)

    generated_ids_first_step = model.generate(
        input_ids_batch,
        attention_mask=attention_mask_batch,
        max_new_tokens=1,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=False,
        num_beams=1
    )

    synchronize_device(DEVICE)
    t_end_prefill_phase = time.perf_counter()

    avg_util_prefill, avg_mem_prefill = mps_monitor_info.stop()  # 获取占位符

    time_prefill_phase = t_end_prefill_phase - t_start_prefill_phase

    throughput_prefill_phase = total_prompt_tokens / time_prefill_phase

    print(f"  Prefill阶段（及首个token生成）耗时: {time_prefill_phase:.4f} 秒")
    print(f"  Prefill阶段输入Token总数: {total_prompt_tokens}")
    print(f"  Prefill阶段（近似）吞吐量: {throughput_prefill_phase:.2f} tokens/秒")
    if not (avg_util_prefill != avg_util_prefill):  # 检查是否为NaN
        print(f"  (请手动观察此阶段的MPS GPU活动)")

    # 5. 测试完整生成阶段 (MAX_NEW_TOKENS)
    print(f"\n阶段2: 测试完整生成性能 (生成 {MAX_NEW_TOKENS} 个新token)")
    if DEVICE != "cpu":
        del generated_ids_first_step  # 尝试释放内存
        empty_cache_device(DEVICE)
        time.sleep(0.5)

    mps_monitor_info.start()  # 提示用户开始手动观察

    t_start_full_generation = time.perf_counter()
    synchronize_device(DEVICE)

    generated_ids_full = model.generate(
        input_ids_batch,
        attention_mask=attention_mask_batch,
        max_new_tokens=MAX_NEW_TOKENS,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=False,
        num_beams=1
    )

    synchronize_device(DEVICE)
    t_end_full_generation = time.perf_counter()

    avg_util_full_gen, avg_mem_full_gen = mps_monitor_info.stop()  # 获取占位符

    time_full_generation = t_end_full_generation - t_start_full_generation
    actual_new_tokens_per_request_full = generated_ids_full.shape[1] - actual_prompt_tokens_per_request
    total_new_tokens_generated_full = actual_new_tokens_per_request_full * NUM_CONCURRENT_REQUESTS

    print(f"  完整生成 ({actual_new_tokens_per_request_full}个新tokens/请求) 耗时: {time_full_generation:.4f} 秒")
    print(f"  完整生成阶段新生成Token总数: {total_new_tokens_generated_full}")
    if not (avg_util_full_gen != avg_util_full_gen):  # 检查是否为NaN
        print(f"  (请手动观察此阶段的MPS GPU活动)")

    # 6. 近似计算Decode阶段性能
    if actual_new_tokens_per_request_full <= 1 and MAX_NEW_TOKENS > 1:
        print("\n  警告: 完整生成阶段实际生成的token数不足以计算后续Decode性能。")
        time_decode_phase_approx = float('nan')
        throughput_decode_phase_approx = float('nan')
        total_decode_phase_tokens = 0
    elif MAX_NEW_TOKENS == 1:
        print("\n  提示: max_new_tokens=1, 没有后续的Decode阶段可供分析。")
        time_decode_phase_approx = float('nan')
        throughput_decode_phase_approx = float('nan')
        total_decode_phase_tokens = 0
    else:
        time_decode_phase_approx = time_full_generation - time_prefill_phase
        total_decode_phase_tokens = total_new_tokens_generated_full - (1 * NUM_CONCURRENT_REQUESTS)

        if time_decode_phase_approx <= 0 or total_decode_phase_tokens <= 0:
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
    print(f"模型: {MODEL_NAME}, 并发数: {NUM_CONCURRENT_REQUESTS}, 设备: {DEVICE}")
    print(
        f"输入长度/请求 (目标): ~{TARGET_PROMPT_TOKENS} tokens, (实际padding后): {actual_prompt_tokens_per_request} tokens")
    print(f"生成长度/请求 (目标): {MAX_NEW_TOKENS} tokens, (实际): {actual_new_tokens_per_request_full} tokens")

    print("\n性能对比:")
    print(f"  1. 近似Prefill阶段 (处理输入 {total_prompt_tokens} tokens 并生成首批 {NUM_CONCURRENT_REQUESTS} tokens):")
    print(f"     - 耗时: {time_prefill_phase:.4f} s")
    print(f"     - 输入吞吐量: {throughput_prefill_phase:.2f} input tokens/s")
    if DEVICE == "mps": print(f"     - (请回顾此阶段手动观察的MPS GPU活动)")

    print(f"\n  2. 近似后续Decode阶段 (生成后续 {total_decode_phase_tokens} tokens):")
    print(f"     - 耗时: {time_decode_phase_approx:.4f} s")
    print(f"     - 输出吞吐量: {throughput_decode_phase_approx:.2f} output tokens/s")
    if DEVICE == "mps": print(f"     - (请回顾此阶段手动观察的MPS GPU活动)")

    print("\n分析:")
    # 检查throughput_decode_phase_approx是否为NaN，如果是，则不能直接比较
    can_compare = not (throughput_decode_phase_approx != throughput_decode_phase_approx) and \
                  not (throughput_prefill_phase != throughput_prefill_phase)

    if can_compare and throughput_decode_phase_approx < throughput_prefill_phase:
        ratio = throughput_prefill_phase / throughput_decode_phase_approx if throughput_decode_phase_approx > 0 else float(
            'inf')
        print(f"  观察到Decode阶段的输出吞吐量 ({throughput_decode_phase_approx:.2f} tokens/s) "
              f"显著低于Prefill阶段的输入吞吐量 ({throughput_prefill_phase:.2f} tokens/s)。")
        print(f"  Prefill吞吐量约是Decode吞吐量的 {ratio:.2f} 倍。")
        print("  这表明，在Decode阶段，即使GPU（在MPS上通过活动监视器观察）可能仍然保持一定的活动，"
              "但每个token的处理速度变慢了。这主要是因为：")
        print("    a. 自回归特性：每个token的生成依赖于前一个，限制了单序列内的并行度。")
        print("    b. 内存带宽限制：频繁读写KV Cache可能成为瓶颈，尤其当上下文变长、并发请求多时。"
              "      (在MPS上，统一内存架构的特性可能与独立显存的NVIDIA GPU表现不同，但带宽仍是因素)")
        print("    c. 计算密度较低：相比Prefill阶段大规模处理输入，Decode阶段每个step的计算量相对较小。")
        print("  这种效率差异正是PD（Prefill-Decode）分离优化所要解决的核心问题，")
        print("  旨在提高Decode阶段的效率，从而提升整体的LLM服务吞吐和响应速度。")
    elif can_compare:
        print("  Decode阶段吞吐量与Prefill阶段吞吐量差异不明显或更高。这可能是由于：")
        print("    - 模型非常小，或者输入/输出序列非常短，导致性能差异不突出。")
        print("    - 特定的并发数和硬件组合导致了不同的瓶颈表现。")
        print("    - MPS的统一内存架构可能使得P/D阶段的内存访问模式差异不像独立显存那样显著。")
        print("    - 近似测量方法引入的误差。")
        print("  请尝试增大模型、序列长度或调整并发数以观察更典型的现象。")
    else:
        print("  未能有效计算吞吐量或进行比较 (可能是因为生成token数过少)。")

    # 清理资源
    del model
    del tokenizer
    if DEVICE != "cpu":
        empty_cache_device(DEVICE)
    print("\n实验结束，资源已清理。")


if __name__ == "__main__":
    if DEVICE == "cpu" and input("警告: 将在CPU上运行，可能非常慢。是否继续? (y/n): ").lower() != 'y':
        print("已取消实验。")
    else:
        run_experiment()