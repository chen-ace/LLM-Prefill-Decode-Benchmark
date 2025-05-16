import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import copy # 用于深拷贝KV Cache结构 (虽然在 transfer_kv_cache 中没有直接使用)
import gc # 用于显式垃圾回收

# --- 配置参数 ---
MODEL_NAME = "gpt2"  # 你可以更改为其他模型，如 "gpt2-medium", "meta-llama/Llama-2-7b-hf" (如果你有权限且VRAM足够)
NUM_CONCURRENT_REQUESTS = 2  # 并发请求的批处理大小
TARGET_PROMPT_TOKENS = 512   # 目标输入Token数量
MAX_NEW_TOKENS = 512         # 最大生成Token数量

# --- 设备检测 ---
DEVICE_GPU = "cpu" # 默认为 CPU
if torch.cuda.is_available():
    DEVICE_GPU = "cuda"
    print(f"检测到 CUDA。将使用 '{DEVICE_GPU}' 进行Prefill和部分Decode场景。")
elif torch.backends.mps.is_available():
    DEVICE_GPU = "mps"
    print(f"检测到 Apple Silicon MPS 后端。将使用 '{DEVICE_GPU}' 进行Prefill和部分Decode场景。")
    print("注意：此脚本主要为CUDA适配，MPS的使用可能具有不同的性能特征。")
else:
    print(f"警告：未检测到 CUDA 或 MPS。Prefill将在CPU上运行 ('{DEVICE_GPU}')。")

DEVICE_CPU = "cpu"
print(f"用于对比Decode阶段的CPU设备：'{DEVICE_CPU}'")


# --- GPU监控提示信息 ---
class GPUMonitorInfo:
    def __init__(self, device_name):
        self.is_cuda_device = (device_name == "cuda")
        self.is_mps_device = (device_name == "mps")

        if self.is_cuda_device:
            print("-" * 30)
            print("CUDA GPU活动监控提示:")
            print("你可以在终端使用 'nvidia-smi' 命令来监控GPU活动。")
            print("例如: `watch -n 0.5 nvidia-smi`")
            print("-" * 30)
        elif self.is_mps_device:
            print("-" * 30)
            print("Apple Silicon (MPS) GPU活动监控提示:")
            print("1. 打开“活动监视器” -> 菜单栏“窗口” -> “GPU历史记录”。")
            print("2. 在终端运行: sudo powermetrics --samplers gpu_power -i 1000")
            print("-" * 30)

    def start(self):
        if self.is_cuda_device:
            print("提示：请按需使用 nvidia-smi 开始观察 CUDA GPU 活动。")
        elif self.is_mps_device:
            print("提示：请手动开始观察 MPS GPU 活动。")

    def stop(self):
        # 对于CUDA, nvidia-smi是一个外部工具, Python无法直接停止
        # 对于MPS, 监控也是手动的
        if self.is_cuda_device:
            print("提示：你可以手动停止观察 nvidia-smi。")
        elif self.is_mps_device:
            print("提示：请手动结束观察 MPS GPU 活动。")
        return float('nan'), float('nan') # 占位符, 实际指标来自外部工具


# --- 辅助函数 ---
def generate_prompt_batch(tokenizer, target_token_count, num_prompts):
    # 使用简单的重复文本来达到目标token数量
    base_text = "请基于以下内容进行详细阐述和扩展："
    base_tokens = tokenizer.encode(base_text, add_special_tokens=False)
    if not base_tokens: # 如果分词器对base_text返回空，则使用后备方案
        base_tokens = tokenizer.encode("测试 ", add_special_tokens=False) or [tokenizer.eos_token_id or 0] # 确保至少有一个token

    num_repeats = (target_token_count + len(base_tokens) - 1) // len(base_tokens)
    single_prompt_tokens = (base_tokens * num_repeats)[:target_token_count]
    single_prompt_text = tokenizer.decode(single_prompt_tokens)
    return [single_prompt_text] * num_prompts


def synchronize_device(device_name):
    if device_name == "cuda":
        torch.cuda.synchronize()
    elif device_name == "mps":
        torch.mps.synchronize()
    # CPU的同步通常对于操作是隐式的


def empty_cache_device(device_name):
    if device_name == "cuda":
        torch.cuda.empty_cache()
    elif device_name == "mps":
        # torch.mps.empty_cache() # 当前PyTorch版本中不可用
        gc.collect() # 通用Python垃圾回收
    gc.collect()


def transfer_kv_cache(past_key_values, target_device):
    if past_key_values is None:
        return None
    transferred_kv = []
    for layer_past in past_key_values:
        # 每个 layer_past 是一个 (key_states, value_states) 的元组
        transferred_kv.append(tuple(tensor.to(target_device) for tensor in layer_past))
    return tuple(transferred_kv)


# --- 主实验逻辑 ---
def run_experiment():
    print(
        f"实验配置: 模型={MODEL_NAME}, 并发请求数={NUM_CONCURRENT_REQUESTS}, "
        f"输入长度约={TARGET_PROMPT_TOKENS}, 生成长度={MAX_NEW_TOKENS}")
    print(f"Prefill设备={DEVICE_GPU}, Decode对比: {DEVICE_GPU} vs {DEVICE_CPU}")

    # 1. 加载模型和分词器
    print("正在加载模型和分词器...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token # 常见做法
            tokenizer.pad_token_id = tokenizer.eos_token_id
            print(f"已将 pad_token 设置为 eos_token ('{tokenizer.pad_token}')")

        print(f"正在为设备 '{DEVICE_GPU}' 加载模型...")
        model_gpu = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE_GPU)
        model_gpu.eval()

        print(f"正在为设备 '{DEVICE_CPU}' 加载模型...")
        if DEVICE_GPU == DEVICE_CPU:
            model_cpu = model_gpu # 如果GPU实际上是CPU，则共享模型
            print(f"   CPU模型与{DEVICE_GPU}模型是同一实例 (两者均为CPU)。")
        else:
            # 如果GPU是CUDA/MPS，则为CPU加载单独的实例
            model_cpu = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE_CPU)
            model_cpu.eval()

    except Exception as e:
        print(f"加载模型或分词器失败: {e}")
        print("如果使用像Llama这样的大模型, 请确保有足够的显存/内存以及必要的访问权限/认证。")
        return

    # 2. 准备输入数据 (批量)
    prompts_batch = generate_prompt_batch(tokenizer, TARGET_PROMPT_TOKENS, NUM_CONCURRENT_REQUESTS)
    inputs_gpu = tokenizer(prompts_batch, return_tensors="pt", padding=True, truncation=True,
                           max_length=TARGET_PROMPT_TOKENS).to(DEVICE_GPU)
    input_ids_batch_gpu = inputs_gpu.input_ids  # 形状: (batch_size, seq_len)
    actual_prompt_tokens_per_request = input_ids_batch_gpu.shape[1]
    total_prompt_tokens = NUM_CONCURRENT_REQUESTS * actual_prompt_tokens_per_request
    print(
        f"输入数据准备完成。每个请求实际输入token数: {actual_prompt_tokens_per_request}, "
        f"总输入token数: {total_prompt_tokens}")

    gpu_monitor_info = GPUMonitorInfo(DEVICE_GPU)

    # 3. 设备预热 (使用批量数据)
    print("正在进行设备预热...")
    if DEVICE_GPU != "cpu":
        try:
            print(f"正在预热 {DEVICE_GPU} 设备...")
            # 执行几次前向传播
            _ = model_gpu(input_ids_batch_gpu[:, :1], use_cache=False) # 小输入
            synchronize_device(DEVICE_GPU)
            temp_out_gpu = model_gpu(input_ids_batch_gpu, use_cache=True) # 类似完整prefill的传递
            temp_kv_gpu = temp_out_gpu.past_key_values
            temp_next_token_gpu = torch.argmax(temp_out_gpu.logits[:, -1:, :], dim=-1)
            _ = model_gpu(temp_next_token_gpu, past_key_values=temp_kv_gpu, use_cache=True) # 类似decode的传递
            synchronize_device(DEVICE_GPU)
            del temp_out_gpu, temp_kv_gpu, temp_next_token_gpu
            empty_cache_device(DEVICE_GPU)
            print(f"{DEVICE_GPU} 设备预热完成。")
        except Exception as e:
            print(f"{DEVICE_GPU} 设备预热期间发生错误: {e}"); return

    if DEVICE_CPU != DEVICE_GPU: # 仅当CPU与GPU设备不同时才单独预热CPU
        try:
            print(f"正在预热 {DEVICE_CPU} 设备...")
            inputs_cpu_warmup = tokenizer(prompts_batch, return_tensors="pt", padding=True,
                                          max_length=TARGET_PROMPT_TOKENS, truncation=True).to(DEVICE_CPU)
            _ = model_cpu(inputs_cpu_warmup.input_ids[:, :1], use_cache=False)
            synchronize_device(DEVICE_CPU) # 对CPU来说不那么关键
            temp_out_cpu = model_cpu(inputs_cpu_warmup.input_ids, use_cache=True)
            temp_kv_cpu = temp_out_cpu.past_key_values
            temp_next_token_cpu = torch.argmax(temp_out_cpu.logits[:, -1:, :], dim=-1)
            _ = model_cpu(temp_next_token_cpu, past_key_values=temp_kv_cpu, use_cache=True)
            synchronize_device(DEVICE_CPU)
            del temp_out_cpu, temp_kv_cpu, temp_next_token_cpu, inputs_cpu_warmup
            empty_cache_device(DEVICE_CPU)
            print(f"{DEVICE_CPU} 设备预热完成。")
        except Exception as e:
            print(f"{DEVICE_CPU} 设备预热期间发生错误: {e}"); return
    else:
        print(f"{DEVICE_CPU} 设备与 {DEVICE_GPU} 相同, 无需单独预热。")


    print("\n--- 开始正式测试 ---")

    # --- Prefill阶段 (在DEVICE_GPU上执行, 批量处理) ---
    print(f"\n阶段 P: 在 {DEVICE_GPU} 上执行Prefill (批量处理)...")
    gpu_monitor_info.start()
    t_start_prefill = time.perf_counter()
    synchronize_device(DEVICE_GPU) # 确保计时准确开始
    with torch.no_grad():
        outputs_prefill_gpu = model_gpu(input_ids_batch_gpu, use_cache=True)
    logits_prefill_gpu = outputs_prefill_gpu.logits  # 形状: (batch_size, seq_len, vocab_size)
    past_key_values_from_gpu_prefill = outputs_prefill_gpu.past_key_values  # 批量的KV cache
    next_token_ids_batch_gpu = torch.argmax(logits_prefill_gpu[:, -1:, :], dim=-1)  # 形状: (batch_size, 1)
    synchronize_device(DEVICE_GPU) # 确保所有操作完成
    t_end_prefill = time.perf_counter()
    gpu_monitor_info.stop()
    time_prefill_gpu = t_end_prefill - t_start_prefill
    throughput_prefill_gpu = total_prompt_tokens / time_prefill_gpu if time_prefill_gpu > 0 else float('inf')
    print(f"  Prefill (总共 {total_prompt_tokens} tokens) 耗时: {time_prefill_gpu:.4f} 秒")
    print(f"  Prefill 吞吐量: {throughput_prefill_gpu:.2f} tokens/秒")

    # --- 场景 A: Decode阶段在 DEVICE_GPU 上执行 (批量处理) ---
    print(f"\n场景 A: Decode阶段在 {DEVICE_GPU} 上执行 (批量处理)...")
    current_input_ids_batch_gpu_decode = next_token_ids_batch_gpu.clone() # 从prefill的下一个token开始
    # 确保KV cache在此场景的正确设备上 (如果prefill在DEVICE_GPU上，它应该已经是了)
    current_past_key_values_batch_gpu_decode = transfer_kv_cache(past_key_values_from_gpu_prefill, DEVICE_GPU)

    gpu_monitor_info.start()
    t_start_decode_gpu = time.perf_counter()
    synchronize_device(DEVICE_GPU)

    total_generated_count_gpu = 0
    with torch.no_grad():
        for i in range(MAX_NEW_TOKENS):
            outputs_decode_gpu = model_gpu(
                input_ids=current_input_ids_batch_gpu_decode,  # (batch_size, 1)
                past_key_values=current_past_key_values_batch_gpu_decode,
                use_cache=True
            )
            logits_decode_gpu = outputs_decode_gpu.logits    # (batch_size, 1, vocab_size)
            current_past_key_values_batch_gpu_decode = outputs_decode_gpu.past_key_values
            current_input_ids_batch_gpu_decode = torch.argmax(logits_decode_gpu, dim=-1)  # (batch_size, 1)
            total_generated_count_gpu += NUM_CONCURRENT_REQUESTS
            if (i + 1) % 100 == 0: # 可选: 打印进度
                 print(f"   已在 {DEVICE_GPU} 上生成 {total_generated_count_gpu} tokens...")


    synchronize_device(DEVICE_GPU)
    t_end_decode_gpu = time.perf_counter()
    gpu_monitor_info.stop()
    time_decode_gpu = t_end_decode_gpu - t_start_decode_gpu
    num_actually_generated_gpu = total_generated_count_gpu
    throughput_decode_gpu = num_actually_generated_gpu / time_decode_gpu if time_decode_gpu > 0 else float('inf')
    print(f"  Decode (总共生成 {num_actually_generated_gpu} tokens) 耗时: {time_decode_gpu:.4f} 秒")
    print(f"  Decode 吞吐量 ({DEVICE_GPU}): {throughput_decode_gpu:.2f} tokens/秒")

    # 如果GPU不是CPU，清理场景A的GPU内存
    if DEVICE_GPU != "cpu":
        del current_input_ids_batch_gpu_decode, current_past_key_values_batch_gpu_decode, outputs_decode_gpu, logits_decode_gpu
        empty_cache_device(DEVICE_GPU)


    # --- 场景 B: Decode阶段在 DEVICE_CPU 上执行 (批量处理, KV Cache来自DEVICE_GPU) ---
    print(f"\n场景 B: Decode阶段在 {DEVICE_CPU} 上执行 (批量处理, KV Cache来自 {DEVICE_GPU})...")

    print(f"  正在将KV Cache从 {DEVICE_GPU} 转移到 {DEVICE_CPU}...")
    t_start_transfer = time.perf_counter()
    # 如果之后打算重用原始GPU KV cache，则在传输前深拷贝past_key_values
    # 对于此脚本，我们已完成此特定KV cache的GPU版本，因此直接传输即可。
    current_past_key_values_batch_cpu = transfer_kv_cache(past_key_values_from_gpu_prefill, DEVICE_CPU)
    # 同时传输初始的next_token_ids
    current_input_ids_batch_cpu = next_token_ids_batch_gpu.to(DEVICE_CPU) # 形状: (batch_size, 1)
    synchronize_device(DEVICE_CPU) # 如果目标也是可同步设备，确保传输完成 (虽然CPU通常不需要)
    t_end_transfer = time.perf_counter()
    time_kv_transfer = t_end_transfer - t_start_transfer
    print(f"  KV Cache和初始tokens转移耗时: {time_kv_transfer:.4f} 秒")

    t_start_decode_cpu = time.perf_counter()
    # CPU计时前无需显式同步，操作通常是阻塞的
    total_generated_count_cpu = 0
    with torch.no_grad():
        for i in range(MAX_NEW_TOKENS):
            outputs_decode_cpu = model_cpu(
                input_ids=current_input_ids_batch_cpu,  # (batch_size, 1)
                past_key_values=current_past_key_values_batch_cpu,
                use_cache=True
            )
            logits_decode_cpu = outputs_decode_cpu.logits      # (batch_size, 1, vocab_size)
            current_past_key_values_batch_cpu = outputs_decode_cpu.past_key_values
            current_input_ids_batch_cpu = torch.argmax(logits_decode_cpu, dim=-1)  # (batch_size, 1)
            total_generated_count_cpu += NUM_CONCURRENT_REQUESTS
            if (i + 1) % 100 == 0: # 可选: 打印进度
                 print(f"   已在 {DEVICE_CPU} 上生成 {total_generated_count_cpu} tokens...")


    # CPU计时后无需显式同步
    t_end_decode_cpu = time.perf_counter()
    time_decode_cpu_compute = t_end_decode_cpu - t_start_decode_cpu
    num_actually_generated_cpu = total_generated_count_cpu
    throughput_decode_cpu_compute = num_actually_generated_cpu / time_decode_cpu_compute if time_decode_cpu_compute > 0 else float('inf')
    time_decode_cpu_with_transfer = time_decode_cpu_compute + time_kv_transfer
    throughput_decode_cpu_with_transfer = num_actually_generated_cpu / time_decode_cpu_with_transfer if time_decode_cpu_with_transfer > 0 else float('inf')

    print(f"  Decode (生成 {num_actually_generated_cpu} tokens, 纯计算) 耗时: {time_decode_cpu_compute:.4f} 秒")
    print(f"  Decode 吞吐量 ({DEVICE_CPU}, 纯计算): {throughput_decode_cpu_compute:.2f} tokens/秒")
    print(f"  Decode (含KV Cache转移) 总耗时: {time_decode_cpu_with_transfer:.4f} 秒")
    print(f"  Decode 吞吐量 ({DEVICE_CPU}, 含KV Cache转移): {throughput_decode_cpu_with_transfer:.2f} tokens/秒")

    # --- 结果总结与分析 ---
    print("\n--- 实验结果总结与分析 ---")
    print(
        f"模型: {MODEL_NAME}, 并发请求数: {NUM_CONCURRENT_REQUESTS}, "
        f"输入长度: {actual_prompt_tokens_per_request}, 每个请求最大生成新Token数: {MAX_NEW_TOKENS}")
    print(f"(GPU实际生成tokens总数: {num_actually_generated_gpu}, CPU实际生成tokens总数: {num_actually_generated_cpu})")

    print(f"\nPrefill阶段 (在{DEVICE_GPU}上, 共{total_prompt_tokens} tokens):")
    print(f"  - 耗时: {time_prefill_gpu:.4f} 秒, 吞吐量: {throughput_prefill_gpu:.2f} tokens/秒")

    total_decode_tokens_expected = NUM_CONCURRENT_REQUESTS * MAX_NEW_TOKENS
    print(f"\nDecode阶段对比 (总共生成约 {total_decode_tokens_expected} tokens):")
    print(f"  场景 A (在{DEVICE_GPU}上Decode):")
    print(f"    - 耗时: {time_decode_gpu:.4f} 秒")
    print(f"    - 吞吐量: {throughput_decode_gpu:.2f} tokens/秒")

    print(f"  场景 B (在{DEVICE_GPU}上Prefill, 在{DEVICE_CPU}上Decode):")
    print(f"    - KV Cache转移耗时 ({DEVICE_GPU} -> {DEVICE_CPU}): {time_kv_transfer:.4f} 秒")
    print(f"    - Decode纯计算耗时 ({DEVICE_CPU}): {time_decode_cpu_compute:.4f} 秒")
    print(f"    - Decode纯计算吞吐量 ({DEVICE_CPU}): {throughput_decode_cpu_compute:.2f} tokens/秒")
    print(f"    - Decode总耗时 (含转移): {time_decode_cpu_with_transfer:.4f} 秒")
    print(f"    - Decode总吞吐量 (含转移): {throughput_decode_cpu_with_transfer:.2f} tokens/秒")

    print("\n分析:")
    print(f"  1. 在{DEVICE_GPU}上的Prefill ({throughput_prefill_gpu:.2f} t/s) 预计会显著快于在CPU上运行，"
          f"特别是在批处理较大和提示较长的情况下，这得益于GPU的并行处理能力。")

    can_compare_gpu_cpu_decode = not (throughput_decode_gpu != throughput_decode_gpu) and \
                               not (throughput_decode_cpu_with_transfer != throughput_decode_cpu_with_transfer) # 检查是否为NaN

    if can_compare_gpu_cpu_decode and throughput_decode_gpu > 0 and throughput_decode_cpu_with_transfer > 0 :
        print(f"  2. 在{DEVICE_GPU}上Decode ({throughput_decode_gpu:.2f} t/s) vs. 在{DEVICE_CPU}上Decode (含转移, {throughput_decode_cpu_with_transfer:.2f} t/s):")
        if throughput_decode_gpu > throughput_decode_cpu_with_transfer:
            ratio_gpu_vs_cpu = throughput_decode_gpu / throughput_decode_cpu_with_transfer
            print(f"     GPU Decode约比CPU Decode (含转移) 快 {ratio_gpu_vs_cpu:.2f} 倍。")
            print(f"     这突显了GPU在逐个token生成方面的优势，即使是批量处理。")
        else:
            ratio_cpu_vs_gpu = throughput_decode_cpu_with_transfer / throughput_decode_gpu
            print(f"     CPU Decode (含转移) 约比GPU Decode快 {ratio_cpu_vs_gpu:.2f} 倍或与之相当。")
            print(f"     这可能发生在模型非常小、批处理量小，或者{DEVICE_GPU}是MPS（其小操作特性可能与CUDA不同）的情况下。")
            print(f"     KV cache转移开销 ({time_kv_transfer:.4f}秒) 是P(GPU)-D(CPU)场景的关键因素。")

        print(f"     纯CPU decode计算吞吐量为 {throughput_decode_cpu_compute:.2f} t/s。")
        print(f"     如果 {time_kv_transfer:.4f}秒 占 {time_decode_cpu_with_transfer:.4f}秒 的重要部分, "
              f"这表明数据移动可能是一个瓶颈。")
    else:
        print("  2. 未能有效比较GPU和CPU的Decode阶段吞吐量 (可能是由于耗时为零或NaN值)。")

    print("  3. P(GPU)-D(CPU)方案的可行性:")
    print("     - 如果CPU的decode速度对于应用来说“足够快”，并且GPU能更快地被释放用于更多的prefill操作，那么这种设置是可行的。")
    print("     - 这是一种权衡：GPU用于decode的时间 vs. KV cache转移开销 + CPU decode时间。")
    print("     - 对于拥有大量并发用户，且TTFT（首个token生成时间，主要由prefill决定）至关重要，而后续在CPU上的token生成速度可接受的系统，"
          "这可能是一种最大化GPU用于prefill任务利用率的策略。")
    print("     - 然而，如果每个用户的端到端生成速度至关重要，并且GPU本身不是瓶颈，那么将decode保留在GPU上通常更快。")

    print(f"  4. 增加 NUM_CONCURRENT_REQUESTS (批处理大小) 通常会提高GPU上prefill和decode的吞吐量，"
          f"直到设备饱和或内存成为限制因素。")


    # 清理
    print("\n正在清理模型并释放内存...")
    del model_gpu, model_cpu, tokenizer
    del inputs_gpu, input_ids_batch_gpu, logits_prefill_gpu, past_key_values_from_gpu_prefill, next_token_ids_batch_gpu
    del current_past_key_values_batch_cpu, current_input_ids_batch_cpu # logits_decode_cpu, outputs_decode_cpu 已在循环作用域内
    empty_cache_device(DEVICE_GPU)
    empty_cache_device(DEVICE_CPU) # 通用 gc.collect()
    print("实验结束。")


if __name__ == "__main__":
    if DEVICE_GPU == "cpu" and DEVICE_CPU == "cpu":
        if input("警告: Prefill和Decode都将在CPU上运行。是否继续? (y/n): ").lower() != 'y':
            print("已取消实验。")
        else:
            run_experiment()
    else:
        run_experiment()