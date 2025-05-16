import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import copy  # 用于深拷贝KV Cache结构

# --- 配置参数 ---
MODEL_NAME = "gpt2"
NUM_CONCURRENT_REQUESTS = 2  # <--- 重新启用并发请求数 (批处理大小)
TARGET_PROMPT_TOKENS = 512
MAX_NEW_TOKENS = 512

# --- 设备检测 ---
if torch.backends.mps.is_available():
    DEVICE_MPS = "mps"
    print("检测到Apple Silicon MPS 后端可用，将使用 'mps' 设备进行Prefill和部分Decode。")
elif torch.cuda.is_available():
    DEVICE_MPS = "cuda"
    print("警告：未检测到MPS，但检测到CUDA。脚本主要为MPS设计，但部分逻辑可能在CUDA上运行。")
else:
    DEVICE_MPS = "cpu"
    print("警告：未检测到MPS或CUDA。Prefill将尝试在CPU上运行。")

DEVICE_CPU = "cpu"
print(f"CPU设备将用于对比Decode阶段：'{DEVICE_CPU}'")


# --- MPS GPU监控提示 ---
class MPSGPUMonitorInfo:
    def __init__(self, device_name):
        self.is_mps_device = (device_name == "mps")
        if self.is_mps_device:
            print("-" * 30)
            print("Apple Silicon (MPS) GPU活动监控提示:")
            # ... (提示内容与之前脚本相同) ...
            print("1. 打开“活动监视器” -> 菜单栏“窗口” -> “GPU历史记录”。")
            print("2. 在终端运行: sudo powermetrics --samplers gpu_power -i 1000")
            print("-" * 30)

    def start(self):
        if self.is_mps_device: print("提示：请手动开始观察MPS GPU活动。")

    def stop(self):
        if self.is_mps_device: print("提示：请手动结束观察MPS GPU活动。")
        return float('nan'), float('nan')


# --- 辅助函数 ---
def generate_prompt_batch(tokenizer, target_token_count, num_prompts):  # 改回批处理版本
    base_text = "请基于以下内容进行详细阐述和扩展："
    base_tokens = tokenizer.encode(base_text, add_special_tokens=False)
    if not base_tokens: base_tokens = tokenizer.encode("测试", add_special_tokens=False) or [0]
    num_repeats = (target_token_count + len(base_tokens) - 1) // len(base_tokens)
    single_prompt_tokens = (base_tokens * num_repeats)[:target_token_count]
    single_prompt_text = tokenizer.decode(single_prompt_tokens)
    return [single_prompt_text] * num_prompts  # 返回批量的prompt


def synchronize_device(device_name):
    if device_name == "cuda":
        torch.cuda.synchronize()
    elif device_name == "mps":
        torch.mps.synchronize()


def empty_cache_device(device_name):
    if device_name == "cuda":
        torch.cuda.empty_cache()
    elif device_name == "mps":
        import gc
        gc.collect()


def transfer_kv_cache(past_key_values, target_device):
    if past_key_values is None: return None
    transferred_kv = []
    for layer_past in past_key_values:
        transferred_kv.append(tuple(tensor.to(target_device) for tensor in layer_past))
    return tuple(transferred_kv)


# --- 主实验逻辑 ---
def run_experiment():
    print(
        f"实验配置: 模型={MODEL_NAME}, 并发数={NUM_CONCURRENT_REQUESTS}, 输入长度约={TARGET_PROMPT_TOKENS}, 生成长度={MAX_NEW_TOKENS}")
    print(f"Prefill设备={DEVICE_MPS}, Decode对比设备={DEVICE_MPS} vs {DEVICE_CPU}")

    # 1. 加载模型和分词器
    print("正在加载模型和分词器...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        print(f"为设备 '{DEVICE_MPS}' 加载模型...")
        model_mps = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE_MPS)
        model_mps.eval()
        print(f"为设备 '{DEVICE_CPU}' 加载模型...")
        if DEVICE_MPS == DEVICE_CPU:
            model_cpu = model_mps
        else:
            model_cpu = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE_CPU)
            model_cpu.eval()
    except Exception as e:
        print(f"加载模型或分词器失败: {e}")
        return

    if tokenizer.pad_token is None: tokenizer.pad_token_id = tokenizer.eos_token_id

    # 2. 准备输入数据 (批量)
    prompts_batch = generate_prompt_batch(tokenizer, TARGET_PROMPT_TOKENS, NUM_CONCURRENT_REQUESTS)
    inputs_mps = tokenizer(prompts_batch, return_tensors="pt", padding=True, truncation=True,
                           max_length=TARGET_PROMPT_TOKENS).to(DEVICE_MPS)
    input_ids_batch_mps = inputs_mps.input_ids  # Shape: (batch_size, seq_len)
    actual_prompt_tokens_per_request = input_ids_batch_mps.shape[1]
    total_prompt_tokens = NUM_CONCURRENT_REQUESTS * actual_prompt_tokens_per_request
    print(
        f"输入数据准备完成。每个请求实际输入token数: {actual_prompt_tokens_per_request}, 总输入: {total_prompt_tokens} tokens")

    mps_monitor_info = MPSGPUMonitorInfo(DEVICE_MPS)

    # 3. 设备预热 (使用批处理数据)
    print("正在进行设备预热...")
    if DEVICE_MPS != "cpu":
        try:
            _ = model_mps(input_ids_batch_mps[:, :1], use_cache=False)
            synchronize_device(DEVICE_MPS)
            temp_out = model_mps(input_ids_batch_mps, use_cache=True)
            temp_kv = temp_out.past_key_values
            temp_next_token = torch.argmax(temp_out.logits[:, -1:, :], dim=-1)  # Shape: (batch_size, 1)
            model_mps(temp_next_token, past_key_values=temp_kv, use_cache=True)
            synchronize_device(DEVICE_MPS)
            del temp_out, temp_kv, temp_next_token
            empty_cache_device(DEVICE_MPS)
            print(f"{DEVICE_MPS} 设备预热完成。")
        except Exception as e:
            print(f"{DEVICE_MPS} 设备预热期间发生错误: {e}"); return
    if DEVICE_CPU != DEVICE_MPS:
        try:
            inputs_cpu_warmup = tokenizer(prompts_batch, return_tensors="pt", padding=True).to(DEVICE_CPU)
            _ = model_cpu(inputs_cpu_warmup.input_ids[:, :1], use_cache=False)
            temp_out_cpu = model_cpu(inputs_cpu_warmup.input_ids, use_cache=True)
            # ... (CPU预热与之前类似，注意batch)
            del temp_out_cpu, inputs_cpu_warmup  # ...
            empty_cache_device(DEVICE_CPU)
            print(f"{DEVICE_CPU} 设备预热完成。")
        except Exception as e:
            print(f"{DEVICE_CPU} 设备预热期间发生错误: {e}")

    print("\n--- 开始正式测试 ---")

    # --- 通用的Prefill阶段 (在MPS上执行，批处理) ---
    print(f"\n阶段 P: 在 {DEVICE_MPS} 上执行Prefill (批处理)...")
    mps_monitor_info.start()
    t_start_prefill = time.perf_counter()
    synchronize_device(DEVICE_MPS)
    with torch.no_grad():
        outputs_prefill_mps = model_mps(input_ids_batch_mps, use_cache=True)
    logits_prefill_mps = outputs_prefill_mps.logits  # Shape: (batch_size, seq_len, vocab_size)
    past_key_values_from_mps_prefill = outputs_prefill_mps.past_key_values  # Batched KV cache
    next_token_ids_batch_mps = torch.argmax(logits_prefill_mps[:, -1:, :], dim=-1)  # Shape: (batch_size, 1)
    synchronize_device(DEVICE_MPS)
    t_end_prefill = time.perf_counter()
    mps_monitor_info.stop()
    time_prefill_mps = t_end_prefill - t_start_prefill
    throughput_prefill_mps = total_prompt_tokens / time_prefill_mps
    print(f"  Prefill ({total_prompt_tokens} tokens total) 耗时: {time_prefill_mps:.4f} 秒")
    print(f"  Prefill 吞吐量: {throughput_prefill_mps:.2f} tokens/秒")

    # --- 场景 A: Decode 阶段在 MPS 上执行 (批处理) ---
    print(f"\n场景 A: Decode 阶段在 {DEVICE_MPS} 上执行 (批处理)...")
    # generated_tokens_mps_batch = [[] for _ in range(NUM_CONCURRENT_REQUESTS)] # 存储每个序列的生成结果

    current_input_ids_batch_mps = next_token_ids_batch_mps.clone()
    current_past_key_values_batch_mps = transfer_kv_cache(past_key_values_from_mps_prefill, DEVICE_MPS)

    mps_monitor_info.start()
    t_start_decode_mps = time.perf_counter()
    synchronize_device(DEVICE_MPS)

    total_generated_count_mps = 0
    with torch.no_grad():
        for _ in range(MAX_NEW_TOKENS):  # 所有序列生成固定长度
            outputs_decode_mps = model_mps(
                input_ids=current_input_ids_batch_mps,  # (batch_size, 1)
                past_key_values=current_past_key_values_batch_mps,
                use_cache=True
            )
            logits_decode_mps = outputs_decode_mps.logits  # (batch_size, 1, vocab_size)
            current_past_key_values_batch_mps = outputs_decode_mps.past_key_values
            current_input_ids_batch_mps = torch.argmax(logits_decode_mps, dim=-1)  # (batch_size, 1)
            total_generated_count_mps += NUM_CONCURRENT_REQUESTS  # 每个step所有请求都生成一个token
            # for i in range(NUM_CONCURRENT_REQUESTS):
            # generated_tokens_mps_batch[i].append(current_input_ids_batch_mps[i, 0].item())

    synchronize_device(DEVICE_MPS)
    t_end_decode_mps = time.perf_counter()
    mps_monitor_info.stop()
    time_decode_mps = t_end_decode_mps - t_start_decode_mps
    # num_actually_generated_mps = sum(len(seq) for seq in generated_tokens_mps_batch)
    num_actually_generated_mps = total_generated_count_mps  # 使用计数器更简单
    throughput_decode_mps = num_actually_generated_mps / time_decode_mps if time_decode_mps > 0 else float('inf')
    print(f"  Decode (总共生成 {num_actually_generated_mps} tokens) 耗时: {time_decode_mps:.4f} 秒")
    print(f"  Decode 吞吐量 ({DEVICE_MPS}): {throughput_decode_mps:.2f} tokens/秒")

    del current_input_ids_batch_mps, current_past_key_values_batch_mps, outputs_decode_mps, logits_decode_mps
    if DEVICE_MPS != "cpu": empty_cache_device(DEVICE_MPS)

    # --- 场景 B: Decode 阶段在 CPU 上执行 (批处理) ---
    print(f"\n场景 B: Decode 阶段在 {DEVICE_CPU} 上执行 (批处理)...")
    # generated_tokens_cpu_batch = [[] for _ in range(NUM_CONCURRENT_REQUESTS)]

    print(f"  正在将KV Cache从 {DEVICE_MPS} 转移到 {DEVICE_CPU}...")
    t_start_transfer = time.perf_counter()
    current_past_key_values_batch_cpu = transfer_kv_cache(past_key_values_from_mps_prefill, DEVICE_CPU)
    current_input_ids_batch_cpu = next_token_ids_batch_mps.to(DEVICE_CPU)  # Shape: (batch_size, 1)
    t_end_transfer = time.perf_counter()
    time_kv_transfer = t_end_transfer - t_start_transfer
    print(f"  KV Cache转移耗时: {time_kv_transfer:.4f} 秒")

    t_start_decode_cpu = time.perf_counter()
    total_generated_count_cpu = 0
    with torch.no_grad():
        for _ in range(MAX_NEW_TOKENS):  # 所有序列生成固定长度
            outputs_decode_cpu = model_cpu(
                input_ids=current_input_ids_batch_cpu,  # (batch_size, 1)
                past_key_values=current_past_key_values_batch_cpu,
                use_cache=True
            )
            logits_decode_cpu = outputs_decode_cpu.logits  # (batch_size, 1, vocab_size)
            current_past_key_values_batch_cpu = outputs_decode_cpu.past_key_values
            current_input_ids_batch_cpu = torch.argmax(logits_decode_cpu, dim=-1)  # (batch_size, 1)
            total_generated_count_cpu += NUM_CONCURRENT_REQUESTS
            # for i in range(NUM_CONCURRENT_REQUESTS):
            #    generated_tokens_cpu_batch[i].append(current_input_ids_batch_cpu[i, 0].item())

    t_end_decode_cpu = time.perf_counter()
    time_decode_cpu = t_end_decode_cpu - t_start_decode_cpu
    # num_actually_generated_cpu = sum(len(seq) for seq in generated_tokens_cpu_batch)
    num_actually_generated_cpu = total_generated_count_cpu
    throughput_decode_cpu = num_actually_generated_cpu / time_decode_cpu if time_decode_cpu > 0 else float('inf')
    time_decode_cpu_with_transfer = time_decode_cpu + time_kv_transfer
    throughput_decode_cpu_with_transfer = num_actually_generated_cpu / time_decode_cpu_with_transfer if time_decode_cpu_with_transfer > 0 else float(
        'inf')

    print(f"  Decode (总共生成 {num_actually_generated_cpu} tokens, 纯计算) 耗时: {time_decode_cpu:.4f} 秒")
    print(f"  Decode 吞吐量 ({DEVICE_CPU}, 纯计算): {throughput_decode_cpu:.2f} tokens/秒")
    print(f"  Decode (含KV Cache转移) 总耗时: {time_decode_cpu_with_transfer:.4f} 秒")
    print(f"  Decode 吞吐量 ({DEVICE_CPU}, 含KV Cache转移): {throughput_decode_cpu_with_transfer:.2f} tokens/秒")

    # --- 结果总结与分析 ---
    print("\n--- 实验结果总结与分析 ---")
    print(
        f"模型: {MODEL_NAME}, 并发数: {NUM_CONCURRENT_REQUESTS}, 输入长度: {actual_prompt_tokens_per_request}, 生成长度: {MAX_NEW_TOKENS}")
    print(f"(MPS实际生成tokens总数: {num_actually_generated_mps}, CPU实际生成tokens总数: {num_actually_generated_cpu})")

    print(f"\nPrefill阶段 (在{DEVICE_MPS}上，共{total_prompt_tokens} tokens):")
    print(f"  - 耗时: {time_prefill_mps:.4f} s, 吞吐量: {throughput_prefill_mps:.2f} tokens/s")

    print(f"\nDecode阶段对比 (共生成约 {NUM_CONCURRENT_REQUESTS * MAX_NEW_TOKENS} tokens):")
    print(f"  场景 A (在{DEVICE_MPS}上):")
    print(f"    - 耗时: {time_decode_mps:.4f} s")
    print(f"    - 吞吐量: {throughput_decode_mps:.2f} tokens/s")

    print(f"  场景 B (在{DEVICE_CPU}上):")
    print(f"    - KV Cache转移耗时: {time_kv_transfer:.4f} s")
    print(f"    - Decode纯计算耗时: {time_decode_cpu:.4f} s")
    print(f"    - Decode纯计算吞吐量: {throughput_decode_cpu:.2f} tokens/s")
    print(f"    - Decode总耗时 (含转移): {time_decode_cpu_with_transfer:.4f} s")
    print(f"    - Decode总吞吐量 (含转移): {throughput_decode_cpu_with_transfer:.2f} tokens/s")

    print("\n分析:")
    print("  1. Prefill阶段在GPU/MPS上通常能通过批处理达到较高吞吐量。")

    can_compare_mps_cpu = not (throughput_decode_mps != throughput_decode_mps) and \
                          not (throughput_decode_cpu_with_transfer != throughput_decode_cpu_with_transfer)

    if can_compare_mps_cpu:
        print(
            f"  2. 对于批处理Decode阶段，MPS的吞吐量 ({throughput_decode_mps:.2f} t/s) 现在应该会显著高于CPU（含转移成本，{throughput_decode_cpu_with_transfer:.2f} t/s）。")
        if throughput_decode_cpu_with_transfer > 0 and throughput_decode_mps > 0:
            ratio_mps_vs_cpu = throughput_decode_mps / throughput_decode_cpu_with_transfer
            print(f"     MPS Decode约比CPU Decode (含转移) 快 {ratio_mps_vs_cpu:.2f} 倍。")
        print("     这是因为批处理为MPS提供了更多的并行工作量，使其能够更有效地利用其计算资源。")
        print("     而CPU虽然也能批处理，但其核心数和并行能力远不及GPU/MPS。")
        print(
            "     如果CPU decode吞吐量仍然意外地高或接近MPS，可能仍与模型非常小、MPS后端对特定小批量操作的优化程度有关。")
    else:
        print("  2. 未能有效计算吞吐量以进行MPS与CPU的Decode阶段比较。")
    print("  3. 随着并发数（批处理大小）的增加，MPS（或传统GPU）相对于CPU的性能优势在Decode阶段通常会更加明显。")
    print("  4. PD分离的可行性：此实验展示了即使在批处理情况下，P和D阶段也可以在不同设备上执行，")
    print("     但性能考量（尤其是数据转移开销和目标设备的处理能力）至关重要。")

    del model_mps, model_cpu, tokenizer
    if DEVICE_MPS != "cpu": empty_cache_device(DEVICE_MPS)
    print("\n实验结束。")


if __name__ == "__main__":
    if DEVICE_MPS == "cpu" and DEVICE_CPU == "cpu" and input(
            "警告: Prefill和Decode都将在CPU上运行。是否继续? (y/n): ").lower() != 'y':
        print("已取消实验。")
    else:
        run_experiment()