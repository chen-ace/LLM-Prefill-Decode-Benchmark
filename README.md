# LLM-Prefill-Decode-Benchmark  

LLM Prefill与Decoding阶段性能对比实验，通过实验对比LLM推理中Prefill和Decoding阶段的吞吐量差异，揭示性能瓶颈，解释PD分离优化技术的原理。包含CUDA和Apple MPS (M系列芯片) 的测试脚本。

## 📖 项目说明

本项目包含两个Python脚本 (`experiment_pd_mps.py` 和 `experiment_pd_cuda.py`)，旨在通过实验直观地展示大型语言模型（LLM）在推理过程中，Prefill（提示词处理）阶段和Decoding（逐Token生成）阶段的性能特性差异。

理解这两个阶段的性能差异是解释为什么需要Prefill/Decoding分离（PD分离）等优化技术的关键。通常情况下：

* **Prefill阶段**：并行处理输入的整个Prompt，计算量相对集中，可以有较高的计算并行度。
* **Decoding阶段**：逐个生成Token，每生成一个Token都依赖于前面已生成的Tokens（通过KV Cache），通常受到显存带宽的限制更为显著。

本实验通过模拟多路并发请求，分别测量计算这两个阶段的吞吐速度（tokens/秒），以揭示它们不同的性能瓶颈。

## 🛠️ 使用方法

### 1. 环境准备

* 确保你的环境已安装 Python 3.x。
* 安装必要的Python库，主要包括 `torch` 和 `transformers`。具体版本请根据你的GPU和CUDA环境选择。
    ```bash
    pip install torch transformers accelerate
    # 根据你的CUDA版本，可能需要特定版本的torch，请参考PyTorch官网
    ```
* 确保你拥有可用的NVIDIA GPU，并已正确安装CUDA驱动程序。

### 2. 下载代码

```bash
git clone https://github.com/chen-ace/LLM-Prefill-Decode-Benchmark.git
cd LLM-Prefill-Decode-Benchmark
```

### 3. 运行实验脚本
你可以选择运行以下任一脚本：

experiment_pd_cuda.py: 一个基于CUDA环境的通用测试脚本。
experiment_pd_mps.py: 此脚本针Apple M系列芯片进行了优化或测试，可以在Apple M系列芯片的系统上运行。

执行命令示例：

```bash
python experiment_pd_cuda.py
```

或者

```bash
python experiment_pd_mps.py
```

脚本默认会模拟一个典型的场景（例如：10路并发，每个请求输入256 tokens，输出256 tokens）。你可能需要根据脚本内的具体实现调整这些参数或模型。

### 4. 观察输出
脚本执行完毕后，会输出Prefill阶段和Decoding阶段各自的吞吐量（tokens/秒）以及可能的GPU利用率等信息。

## 📊 结果解读
实验结果通常会显示以下关键现象：

### 1.Prefill阶段吞吐量显著高于Decoding阶段：

原因：Prefill阶段一次性处理整个输入序列（例如256个tokens），可以利用GPU的大规模并行计算能力，将多个请求的Prefill操作合并成一个大的批次（batch），从而实现较高的tokens处理速度。

表现：你会看到一个相对较高的 Prefill throughput 值。

### 2.Decoding阶段吞吐量发生“断崖式下跌”：

原因：Decoding阶段是逐个token生成的。即使是多路并发（例如10路），每一步也只是为这10路请求各生成一个token。这个过程高度依赖于对KV Cache的读写，而KV Cache会随着生成token数量的增加而增大，对显存带宽的压力也随之增加。计算单元可能并未完全饱和，而是在等待数据IO。

表现：Decoding throughput 值会远低于Prefill阶段的吞吐量。

### 3.GPU计算能力在Decoding阶段未被充分利用：

现象的佐证：吞吐量的断崖式下跌本身就暗示了这一点。


## ⚠️ 重要注意事项：关于GPU-Util指标的解读（尤其在 experiment_pd_cuda.py 中观察时） ⚠️

GPU-Util 这个指标（通常来自 nvidia-smi）表示GPU上有任何类型的工作在执行的时间百分比，包括计算任务、内存操作等。在Decoding阶段，由于频繁且大量的KV Cache读写操作，GPU的内存控制器会非常繁忙，导致GPU-Util指标可能显示得很高。然而，这时的计算核心（CUDA Cores / Tensor Cores）可能因为等待数据而处于空闲或低利用率状态。因此，高GPU-Util在Decoding阶段并不直接等同于高计算效率，反而可能掩盖了计算单元利用率不足的真相，进一步印证了其Memory-Bound（显存带宽受限）的特性。

要更精确地判断计算单元的利用率，需要借助更专业的性能分析工具，如NVIDIA Nsight Systems或Nsight Compute，它们可以提供SM（Streaming Multiprocessor）活跃度、Tensor Core利用率、显存带宽实际使用量等更细致的指标。

## 结论：

通过本实验，我们可以清晰地看到Prefill和Decoding阶段在性能特征上的巨大差异。Prefill阶段倾向于计算密集型（Compute-Bound）或至少可以从大规模并行计算中受益，而Decoding阶段则更倾向于显存带宽密集型（Memory-Bound）。

这种差异正是PD分离（Prefill-Decoding Separation）技术能够提升LLM推理效率的根本原因。通过将这两个阶段分开调度和优化（例如，为Prefill构建大批次，为Decoding优化KV Cache访问和管理），可以更有效地利用GPU资源，从而提高整体吞吐并降低延迟。
