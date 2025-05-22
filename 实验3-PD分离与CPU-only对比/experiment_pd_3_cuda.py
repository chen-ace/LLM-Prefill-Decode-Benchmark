import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import copy  # 用于深拷贝KV Cache结构
import gc  # 用于垃圾回收

# --- 配置参数 ---
MODEL_NAME = "gpt2"
NUM_CONCURRENT_REQUESTS = 100  # 并发请求数 (批处理大小) - 根据你之前的1024token推断，可调整
TARGET_PROMPT_TOKENS = 512  # 每个请求的目标输入Token数
MAX_NEW_TOKENS = 512  # 每个请求的最大生成Token数

# --- 设备检测 ---
if torch.cuda.is_available():
    DEVICE_GPU = "cuda"
    print(f"检测到NVIDIA CUDA 后端可用，将使用 '{DEVICE_GPU}' 设备。")
elif torch.backends.mps.is_available():  # 保留MPS作为备选，以防脚本在Mac上意外运行CUDA版本
    DEVICE_GPU = "mps"
    print(f"警告：未检测到CUDA，但检测到MPS。脚本将使用 '{DEVICE_GPU}' 作为GPU设备 (此版本主要为CUDA设计)。")
else:
    DEVICE_GPU = "cpu"  # Fallback if no GPU at all
    print(f"警告：未检测到CUDA或MPS。GPU相关操作将尝试在 '{DEVICE_GPU}' (CPU)上运行。")

DEVICE_CPU = "cpu"
print(f"CPU设备将用于对比场景：'{DEVICE_CPU}'")


# --- GPU监控提示 ---
class GPUMonitorInfo:  # 从 MPSGPUMonitorInfo 改名
    def __init__(self, device_name):
        self.active_device_name = device_name
        self.is_cuda_device = (device_name == "cuda")
        self.is_mps_device = (device_name == "mps")  # 保留MPS提示逻辑

        if self.is_cuda_device:
            print("-" * 30)
            print("NVIDIA CUDA GPU活动监控提示:")
            print("此脚本不包含自动GPU利用率跟踪。您可以通过以下方式手动观察：")
            print("1. 在终端运行: nvidia-smi")
            print("2. 或更动态地: watch -n 0.1 nvidia-smi")
            print("请在脚本的关键阶段（如模型生成时）观察GPU活动和显存使用。")
            print("-" * 30)
        elif self.is_mps_device:
            # ... (MPS提示内容与之前脚本相同) ...
            print("-" * 30)
            print("Apple Silicon (MPS) GPU活动监控提示:")
            print("1. 打开“活动监视器” -> 菜单栏“窗口” -> “GPU历史记录”。")
            print("2. 在终端运行: sudo powermetrics --samplers gpu_power -i 1000")
            print("-" * 30)

    def start(self, scenario_label=""):
        if self.is_cuda_device:
            print(f"提示 ({scenario_label}): 请手动开始观察CUDA GPU活动 (nvidia-smi)。")
        elif self.is_mps_device:
            print(f"提示 ({scenario_label}): 请手动开始观察MPS GPU活动。")

    def stop(self, scenario_label=""):
        if self.is_cuda_device:
            print(f"提示 ({scenario_label}): 请手动结束观察CUDA GPU活动。")
        elif self.is_mps_device:
            print(f"提示 ({scenario_label}): 请手动结束观察MPS GPU活动。")
        return float('nan'), float('nan')


# --- 辅助函数 ---
def generate_prompt_batch(tokenizer, target_token_count, num_prompts):
    base_text = "请详细解释以下概念并给出具体示例："
    base_tokens = tokenizer.encode(base_text, add_special_tokens=False)
    if not base_tokens: base_tokens = tokenizer.encode("测试", add_special_tokens=False) or [0]
    num_repeats = (target_token_count + len(base_tokens) - 1) // len(base_tokens)
    single_prompt_tokens = (base_tokens * num_repeats)[:target_token_count]
    single_prompt_text = tokenizer.decode(single_prompt_tokens)
    return [single_prompt_text] * num_prompts


def synchronize_device(device_name):
    if device_name == "cuda":
        torch.cuda.synchronize()
    elif device_name == "mps":
        torch.mps.synchronize()


def empty_cache_device(device_name):
    if device_name == "cuda":
        torch.cuda.empty_cache()
    elif device_name == "mps":
        torch.mps.empty_cache() if hasattr(torch.mps, "empty_cache") else None
    gc.collect()


def transfer_kv_cache(past_key_values, target_device):
    if past_key_values is None: return None
    transferred_kv = []
    for layer_past in past_key_values:
        # 对于CUDA到CPU的传输，non_blocking=False是默认且安全的
        transferred_kv.append(tuple(tensor.to(target_device, non_blocking=False) for tensor in layer_past))
    return tuple(transferred_kv)


def manual_decode_loop(model, num_tokens_to_generate, initial_token_ids_batch, initial_past_key_values_batch, device,
                       batch_size):
    current_input_ids_batch = initial_token_ids_batch.to(device)
    current_past_key_values_batch = initial_past_key_values_batch  # 假设已在目标设备

    total_generated_tokens_count = 0
    with torch.no_grad():
        for _ in range(num_tokens_to_generate):
            outputs_decode = model(
                input_ids=current_input_ids_batch,
                past_key_values=current_past_key_values_batch,
                use_cache=True
            )
            logits_decode = outputs_decode.logits
            current_past_key_values_batch = outputs_decode.past_key_values
            current_input_ids_batch = torch.argmax(logits_decode, dim=-1)
            total_generated_tokens_count += batch_size
    return total_generated_tokens_count


def ensure_gpu_ready(model_on_gpu, input_ids_sample_gpu, device_gpu_name,
                     scenario_label):  # ensure_mps_ready -> ensure_gpu_ready
    """在计时前执行小的GPU操作以确保设备就绪"""
    if model_on_gpu and device_gpu_name != "cpu":  # 适用于任何实际的GPU设备
        # print(f"  ({scenario_label} - {device_gpu_name} '唤醒'操作...)")
        with torch.no_grad():
            _ = model_on_gpu(input_ids_sample_gpu[:, :1], use_cache=False)
        synchronize_device(device_gpu_name)
        # print(f"  ({scenario_label} - {device_gpu_name} '唤醒'完成)")


# --- 主实验逻辑 ---
def run_experiment():
    print(
        f"实验配置: 模型={MODEL_NAME}, 并发数={NUM_CONCURRENT_REQUESTS}, 输入长度约={TARGET_PROMPT_TOKENS}, 生成长度={MAX_NEW_TOKENS}")

    results = {}

    print("\n--- 正在加载资源 ---")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token is None: tokenizer.pad_token_id = tokenizer.eos_token_id

        model_gpu = None  # 原 model_mps
        if DEVICE_GPU != "cpu":
            print(f"为设备 '{DEVICE_GPU}' 加载模型...")
            model_gpu = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE_GPU)
            model_gpu.eval()

        print(f"为设备 '{DEVICE_CPU}' 加载模型...")
        model_cpu = model_gpu if DEVICE_GPU == DEVICE_CPU else AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(
            DEVICE_CPU)
        model_cpu.eval()

    except Exception as e:
        print(f"加载模型或分词器失败: {e}"); return

    prompts_batch = generate_prompt_batch(tokenizer, TARGET_PROMPT_TOKENS, NUM_CONCURRENT_REQUESTS)
    # GPU的输入
    inputs_gpu_tokenized = tokenizer(prompts_batch, return_tensors="pt", padding=True, truncation=True,
                                     max_length=TARGET_PROMPT_TOKENS)
    input_ids_batch_gpu = inputs_gpu_tokenized.input_ids.to(DEVICE_GPU)  # 原 input_ids_batch_mps

    input_ids_batch_cpu = None
    if DEVICE_GPU == DEVICE_CPU:
        input_ids_batch_cpu = input_ids_batch_gpu
    else:
        inputs_cpu_tokenized = tokenizer(prompts_batch, return_tensors="pt", padding=True, truncation=True,
                                         max_length=TARGET_PROMPT_TOKENS)
        input_ids_batch_cpu = inputs_cpu_tokenized.input_ids.to(DEVICE_CPU)

    actual_prompt_tokens_per_request = input_ids_batch_gpu.shape[1]
    total_prompt_tokens = NUM_CONCURRENT_REQUESTS * actual_prompt_tokens_per_request
    total_target_generated_tokens = NUM_CONCURRENT_REQUESTS * MAX_NEW_TOKENS
    print(
        f"输入数据准备完成。每个请求实际输入token数: {actual_prompt_tokens_per_request}, 总输入: {total_prompt_tokens} tokens")
    print(f"目标生成token总数: {total_target_generated_tokens} tokens")

    gpu_monitor_info = GPUMonitorInfo(DEVICE_GPU)  # 原 mps_monitor_info

    print("\n--- 正在进行设备预热 ---")
    if model_gpu and DEVICE_GPU != "cpu":
        try:
            ensure_gpu_ready(model_gpu, input_ids_batch_gpu, DEVICE_GPU, "初始预热")
            with torch.no_grad():
                temp_out = model_gpu(input_ids_batch_gpu, use_cache=True)
                temp_kv = temp_out.past_key_values
                temp_next_token = torch.argmax(temp_out.logits[:, -1:, :], dim=-1)
                _ = model_gpu(temp_next_token, past_key_values=temp_kv, use_cache=True)
                synchronize_device(DEVICE_GPU)
            del temp_out, temp_kv, temp_next_token;
            empty_cache_device(DEVICE_GPU)
            print(f"{DEVICE_GPU} 设备预热完成。")
        except Exception as e:
            print(f"{DEVICE_GPU} 设备预热错误: {e}"); return

    if model_cpu:
        try:
            with torch.no_grad():
                _ = model_cpu(input_ids_batch_cpu[:, :min(10, actual_prompt_tokens_per_request)], use_cache=False)
                temp_out_cpu = model_cpu(input_ids_batch_cpu, use_cache=True)
                temp_kv_cpu = temp_out_cpu.past_key_values
                temp_next_token_cpu = torch.argmax(temp_out_cpu.logits[:, -1:, :], dim=-1)
                _ = model_cpu(temp_next_token_cpu, past_key_values=temp_kv_cpu, use_cache=True)
            del temp_out_cpu, temp_kv_cpu, temp_next_token_cpu;
            empty_cache_device(DEVICE_CPU)
            print(f"{DEVICE_CPU} 设备预热完成。")
        except Exception as e:
            print(f"{DEVICE_CPU} 设备预热错误: {e}"); return

    print("\n--- 开始正式测试 ---")

    # --- 场景 A: Prefill on GPU, Decode on GPU (All-GPU) ---
    scenario_A_key = "A_All_GPU"  # Key for results dictionary
    if model_gpu and DEVICE_GPU != "cpu":
        print(f"\n场景 A: Prefill on {DEVICE_GPU}, Decode on {DEVICE_GPU} (All-GPU)")
        gpu_monitor_info.start("场景 A")

        ensure_gpu_ready(model_gpu, input_ids_batch_gpu, DEVICE_GPU, "场景 A Prefill")
        t_start_prefill_A = time.perf_counter();
        synchronize_device(DEVICE_GPU)
        with torch.no_grad():
            outputs_prefill_A_gpu = model_gpu(input_ids_batch_gpu, use_cache=True)
        logits_A_gpu = outputs_prefill_A_gpu.logits
        past_key_values_A_gpu = outputs_prefill_A_gpu.past_key_values
        next_token_ids_A_gpu = torch.argmax(logits_A_gpu[:, -1:, :], dim=-1)
        synchronize_device(DEVICE_GPU);
        t_end_prefill_A = time.perf_counter()
        time_prefill_A = t_end_prefill_A - t_start_prefill_A

        past_key_values_for_A_decode = transfer_kv_cache(past_key_values_A_gpu, DEVICE_GPU)

        ensure_gpu_ready(model_gpu, next_token_ids_A_gpu, DEVICE_GPU, "场景 A Decode")
        t_start_decode_A = time.perf_counter();
        synchronize_device(DEVICE_GPU)
        num_generated_A = manual_decode_loop(model_gpu, MAX_NEW_TOKENS, next_token_ids_A_gpu,
                                             past_key_values_for_A_decode, DEVICE_GPU, NUM_CONCURRENT_REQUESTS)
        synchronize_device(DEVICE_GPU);
        t_end_decode_A = time.perf_counter()
        time_decode_A = t_end_decode_A - t_start_decode_A
        gpu_monitor_info.stop("场景 A")

        results[scenario_A_key] = {
            "description": f"All-{DEVICE_GPU}",
            "time_prefill": time_prefill_A,
            "tput_prefill": total_prompt_tokens / time_prefill_A if time_prefill_A > 0 else 0,
            "time_decode": time_decode_A, "tput_decode": num_generated_A / time_decode_A if time_decode_A > 0 else 0,
            "num_generated_decode": num_generated_A,
            "time_total": time_prefill_A + time_decode_A,
            "tput_overall": (total_prompt_tokens + num_generated_A) / (time_prefill_A + time_decode_A) if (
                                                                                                                      time_prefill_A + time_decode_A) > 0 else 0,
            "time_transfer_kv": 0
        }
        print(f"  Prefill耗时: {time_prefill_A:.4f}s, 吞吐量: {results[scenario_A_key]['tput_prefill']:.2f} t/s")
        print(
            f"  Decode耗时: {time_decode_A:.4f}s, 吞吐量: {results[scenario_A_key]['tput_decode']:.2f} t/s (生成{num_generated_A} tokens)")
        del outputs_prefill_A_gpu, logits_A_gpu, past_key_values_A_gpu, next_token_ids_A_gpu, past_key_values_for_A_decode;
        empty_cache_device(DEVICE_GPU)
    else:
        print(f"\n场景 A: 跳过 (DEVICE_GPU是CPU或模型未加载)")
        results[scenario_A_key] = {"description": f"All-{DEVICE_GPU}", "error": "Skipped"}

    # --- 场景 C: Prefill on GPU, Decode on CPU (PD-Split) ---
    scenario_C_key = "C_P_gpu_D_cpu"  # Key for results dictionary
    if model_gpu and model_cpu and DEVICE_GPU != "cpu" and DEVICE_GPU != DEVICE_CPU:  # Ensure GPU is not CPU and different from CPU device for this scenario
        print(f"\n场景 C: Prefill on {DEVICE_GPU}, Decode on {DEVICE_CPU} (PD-Split)")

        ensure_gpu_ready(model_gpu, input_ids_batch_gpu, DEVICE_GPU, "场景 C Prefill")
        gpu_monitor_info.start("场景 C Prefill")
        t_start_prefill_C = time.perf_counter();
        synchronize_device(DEVICE_GPU)
        with torch.no_grad():
            outputs_prefill_C_gpu = model_gpu(input_ids_batch_gpu, use_cache=True)
        logits_C_gpu = outputs_prefill_C_gpu.logits
        past_key_values_from_gpu_C = outputs_prefill_C_gpu.past_key_values
        next_token_ids_from_gpu_C = torch.argmax(logits_C_gpu[:, -1:, :], dim=-1)
        synchronize_device(DEVICE_GPU);
        t_end_prefill_C = time.perf_counter()
        time_prefill_C = t_end_prefill_C - t_start_prefill_C
        gpu_monitor_info.stop("场景 C Prefill")

        print(f"  正在将KV Cache从 {DEVICE_GPU} 转移到 {DEVICE_CPU}...")
        t_start_transfer_C = time.perf_counter()
        past_key_values_for_cpu_decode_C = transfer_kv_cache(past_key_values_from_gpu_C, DEVICE_CPU)
        next_token_ids_for_cpu_decode_C = next_token_ids_from_gpu_C.to(DEVICE_CPU)
        synchronize_device(DEVICE_GPU)  # Ensure GPU ops (like .to(DEVICE_CPU) if non_blocking) are done
        t_end_transfer_C = time.perf_counter()
        time_transfer_C = t_end_transfer_C - t_start_transfer_C
        print(f"  KV Cache转移耗时: {time_transfer_C:.4f} 秒")

        t_start_decode_C = time.perf_counter()
        num_generated_C = manual_decode_loop(model_cpu, MAX_NEW_TOKENS, next_token_ids_for_cpu_decode_C,
                                             past_key_values_for_cpu_decode_C, DEVICE_CPU, NUM_CONCURRENT_REQUESTS)
        t_end_decode_C = time.perf_counter()
        time_decode_C = t_end_decode_C - t_start_decode_C

        results[scenario_C_key] = {
            "description": f"P({DEVICE_GPU})-D({DEVICE_CPU})",
            "time_prefill": time_prefill_C,
            "tput_prefill": total_prompt_tokens / time_prefill_C if time_prefill_C > 0 else 0,
            "time_transfer_kv": time_transfer_C,
            "time_decode": time_decode_C, "tput_decode": num_generated_C / time_decode_C if time_decode_C > 0 else 0,
            "num_generated_decode": num_generated_C,
            "time_total": time_prefill_C + time_transfer_C + time_decode_C,
            "tput_overall": (total_prompt_tokens + num_generated_C) / (
                        time_prefill_C + time_transfer_C + time_decode_C) if (
                                                                                         time_prefill_C + time_transfer_C + time_decode_C) > 0 else 0
        }
        print(
            f"  Prefill ({DEVICE_GPU})耗时: {time_prefill_C:.4f}s, 吞吐量: {results[scenario_C_key]['tput_prefill']:.2f} t/s")
        print(
            f"  Decode ({DEVICE_CPU})耗时: {time_decode_C:.4f}s, 吞吐量: {results[scenario_C_key]['tput_decode']:.2f} t/s (生成{num_generated_C} tokens)")
        del outputs_prefill_C_gpu, logits_C_gpu, past_key_values_from_gpu_C, next_token_ids_from_gpu_C
        del past_key_values_for_cpu_decode_C, next_token_ids_for_cpu_decode_C
        if DEVICE_GPU != "cpu": empty_cache_device(DEVICE_GPU)
        empty_cache_device(DEVICE_CPU)
    else:
        print(f"\n场景 C: 跳过 (不满足运行条件, 如DEVICE_GPU为CPU或与CPU相同)")
        results[scenario_C_key] = {"description": f"P({DEVICE_GPU})-D({DEVICE_CPU})", "error": "Skipped"}

    # --- 场景 B: Prefill on CPU, Decode on CPU (All-CPU) ---
    scenario_B_key = "B_All_CPU"  # Key for results dictionary
    if model_cpu:
        print(f"\n场景 B: Prefill on {DEVICE_CPU}, Decode on {DEVICE_CPU} (All-CPU)")

        t_start_prefill_B = time.perf_counter()
        with torch.no_grad():
            outputs_prefill_B_cpu = model_cpu(input_ids_batch_cpu, use_cache=True)
        logits_B_cpu = outputs_prefill_B_cpu.logits
        past_key_values_B_cpu = outputs_prefill_B_cpu.past_key_values
        next_token_ids_B_cpu = torch.argmax(logits_B_cpu[:, -1:, :], dim=-1)
        t_end_prefill_B = time.perf_counter()
        time_prefill_B = t_end_prefill_B - t_start_prefill_B

        t_start_decode_B = time.perf_counter()
        num_generated_B = manual_decode_loop(model_cpu, MAX_NEW_TOKENS, next_token_ids_B_cpu, past_key_values_B_cpu,
                                             DEVICE_CPU, NUM_CONCURRENT_REQUESTS)
        t_end_decode_B = time.perf_counter()
        time_decode_B = t_end_decode_B - t_start_decode_B

        results[scenario_B_key] = {
            "description": f"All-{DEVICE_CPU}",
            "time_prefill": time_prefill_B,
            "tput_prefill": total_prompt_tokens / time_prefill_B if time_prefill_B > 0 else 0,
            "time_decode": time_decode_B, "tput_decode": num_generated_B / time_decode_B if time_decode_B > 0 else 0,
            "num_generated_decode": num_generated_B,
            "time_total": time_prefill_B + time_decode_B,
            "tput_overall": (total_prompt_tokens + num_generated_B) / (time_prefill_B + time_decode_B) if (
                                                                                                                      time_prefill_B + time_decode_B) > 0 else 0,
            "time_transfer_kv": 0
        }
        print(f"  Prefill耗时: {time_prefill_B:.4f}s, 吞吐量: {results[scenario_B_key]['tput_prefill']:.2f} t/s")
        print(
            f"  Decode耗时: {time_decode_B:.4f}s, 吞吐量: {results[scenario_B_key]['tput_decode']:.2f} t/s (生成{num_generated_B} tokens)")
        del outputs_prefill_B_cpu, logits_B_cpu, past_key_values_B_cpu, next_token_ids_B_cpu;
        empty_cache_device(DEVICE_CPU)
    else:
        print(f"\n场景 B: 跳过 (CPU模型未加载)")
        results[scenario_B_key] = {"description": f"All-{DEVICE_CPU}", "error": "Skipped"}

    # --- 结果汇总统计 ---
    print("\n--- 实验结果汇总统计 ---")
    # ... (汇总表格打印逻辑与之前脚本相同, 注意使用新的scenario_X_key变量) ...
    print(
        f"模型: {MODEL_NAME}, 并发数: {NUM_CONCURRENT_REQUESTS}, 输入长度/请求: {actual_prompt_tokens_per_request}, 生成长度/请求: {MAX_NEW_TOKENS}")
    print(f"总输入Tokens: {total_prompt_tokens}, 目标总生成Tokens: {total_target_generated_tokens}")
    print("-" * 125)
    header = f"{'场景描述':<25} | {'Prefill Time (s)':<18} | {'Prefill Tput (t/s)':<20} | {'KV Tx Time (s)':<15} | {'Decode Time (s)':<18} | {'Decode Tput (t/s)':<20} | {'Total Time (s)':<18} | {'Overall Tput (t/s)':<20}"
    print(header)
    print("-" * len(header))

    scenario_print_order = [scenario_A_key, scenario_C_key, scenario_B_key]
    for scenario_key in scenario_print_order:
        res = results.get(scenario_key)
        if res and "error" not in res:
            desc = res.get('description', 'N/A')
            t_pref = res.get('time_prefill', float('nan'))
            tp_pref = res.get('tput_prefill', float('nan'))
            t_kvtx = res.get('time_transfer_kv', 0)
            t_dec = res.get('time_decode', float('nan'))
            tp_dec = res.get('tput_decode', float('nan'))
            t_total = res.get('time_total', float('nan'))
            tp_overall = res.get('tput_overall', float('nan'))
            print(
                f"{desc:<25} | {t_pref:<18.4f} | {tp_pref:<20.2f} | {t_kvtx:<15.4f} | {t_dec:<18.4f} | {tp_dec:<20.2f} | {t_total:<18.4f} | {tp_overall:<20.2f}")
        elif res and "error" in res:
            desc = res.get('description', 'N/A')
            print(
                f"{desc:<25} | {'Skipped':<18} | {'Skipped':<20} | {'Skipped':<15} | {'Skipped':<18} | {'Skipped':<20} | {'Skipped':<18} | {'Skipped':<20}")
    print("-" * len(header))

    print("\n分析提示:")
    print("  - Prefill Tput: 输入tokens处理速度。GPU通常远高于CPU。")
    print("  - Decode Tput: 新生成tokens的速度。比较不同设备/策略下的Decode效率。")
    print("  - KV Tx Time: 在PD分离且跨设备时，KV Cache的转移开销。")
    print("  - Overall Tput: (总输入tokens + 总生成tokens) / 总耗时。衡量端到端效率。")

    del model_gpu, model_cpu, tokenizer
    if DEVICE_GPU != "cpu": empty_cache_device(DEVICE_GPU)
    empty_cache_device(DEVICE_CPU)
    print("\n实验结束。")


if __name__ == "__main__":
    run_experiment()
