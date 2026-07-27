# KaReserve 实验流程

## 环境

服务器保留两个独立环境：

```text
.venv-vllm-0.26          vLLM 0.26、Torch、CUDA和Benchmark
.venv-vllm-0.26-lmcache  LMCache Overlay，复用基础环境
```

LMCache Overlay通过 `.pth`引用基础环境。两个目录承担不同依赖职责。

## 模型

`scripts/debug/start_vllm_cluster.sh`通过环境变量选择模型：

```bash
KARESERVE_MODEL=/home/zn/llm_models/opt-1.3b \
KARESERVE_MODEL_NAME=kareserve-opt-1.3b \
KARESERVE_DTYPE=half \
KARESERVE_GPU_MEMORY_UTILIZATION=0.5 \
GPU_IDS=1 \
bash scripts/debug/start_stack.sh
```

当前服务器模型目录包含：

```text
/home/zn/llm_models/opt-1.3b
/home/zn/llm_models/opt-6.7b
/home/zn/llm_models/gpt2
/home/zn/llm_models/gpt2-xl
```

`KARESERVE_CHAT_TEMPLATE`指定Chat Template。空字符串关闭显式模板，模型服务随后使用自身配置。

## 启动

默认单 GPU1、LMCache启用、Router端口8090：

```bash
cd /home/zn/xyz/serve1
GPU_IDS=1 bash scripts/debug/start_stack.sh
```

该脚本依次启动 LMCache MP Server、vLLM实例和 KaReserve Router，并等待 vLLM健康检查。关闭全部项目进程：

```bash
bash scripts/debug/stop_debug_cluster.sh
```

Router配置通过 `KARESERVE_CONFIG_PATH`指定：

```bash
KARESERVE_CONFIG_PATH=/home/zn/xyz/serve1/configs/router.two-node.json \
GPU_IDS="0 1" \
bash scripts/debug/start_stack.sh
```

## 数据集

正式工作负载入口为 `scripts/benchmark/run_vllm_benchmark.sh`。默认数据集是 vLLM内置 `prefix_repetition`，该数据集直接控制共享 Prefix长度、Suffix长度、Prefix数量和输出长度，适合验证 Prefix路由。

```bash
KARESERVE_DATASET_NAME=prefix_repetition \
KARESERVE_PREFIX_LEN=512 \
KARESERVE_SUFFIX_LEN=64 \
KARESERVE_NUM_PREFIXES=8 \
KARESERVE_OUTPUT_LEN=16 \
KARESERVE_NUM_PROMPTS=128 \
KARESERVE_REQUEST_RATE=20 \
bash scripts/benchmark/run_vllm_benchmark.sh
```

外部数据集通过 `KARESERVE_DATASET_NAME`和 `KARESERVE_DATASET_PATH`指定：

```bash
KARESERVE_DATASET_NAME=sharegpt \
KARESERVE_DATASET_PATH=/path/to/sharegpt.json \
bash scripts/benchmark/run_vllm_benchmark.sh
```

服务器协作者数据包括：

```text
/home/zn/datasets/sharegpt             约1.8 GB，ShareGPT JSONL
/home/zn/datasets/Longbench            约459 MB，长上下文JSONL
/home/zn/datasets/arxiv_summarization  约7.1 GB，HuggingFace Arrow
```

ShareGPT文件使用 `conversation`字段的 JSONL格式。协作者脚本 `/home/zn/vllm_sharegpt_loadtest.py`支持该格式。vLLM官方 `sharegpt` Loader的直接兼容性需要单独验证。LongBench和ArXiv需要转换为 vLLM `custom`或 `timed_trace`输入后再用于正式实验。

## 模型与数据的职责边界

Router配置只包含节点地址、KV Event地址和路由参数。模型由 vLLM启动参数指定。数据集由 Benchmark启动参数指定。该结构允许同一 Router配置复用不同模型与数据集。

```text
configs/router.*.json          节点与路由策略
KARESERVE_MODEL                模型路径
KARESERVE_MODEL_NAME           API模型名
KARESERVE_TOKENIZER            Benchmark Tokenizer
KARESERVE_DATASET_NAME         数据集类型
KARESERVE_DATASET_PATH         外部数据路径
```

## 对比对象

策略对比使用相同模型、数据集、请求率、seed和缓存初始状态：

```text
Round Robin       到达顺序轮询
Least Load        只使用实例负载
Prefix Hash       固定Prefix映射
Windowed Prefix   GPU Prefix、负载与容量联合选点
```

策略通过环境变量覆盖：

```bash
KARESERVE_POLICY_OVERRIDE=least_load \
bash scripts/debug/start_router.sh
```

窗口消融使用同一个 `windowed_prefix`策略：

```text
KARESERVE_WINDOW_MS_OVERRIDE=0  单请求立即选点
KARESERVE_WINDOW_MS_OVERRIDE=2  短时窗口联合选点
```

LMCache消融通过 `KARESERVE_ENABLE_LMCACHE=0`和`1`控制。每轮正式实验需要停止整个 Stack并重新启动，使GPU KVCache和LMCache具有一致的冷启动状态。

Benchmark结果默认写入 `runtime/benchmarks/`。`KARESERVE_BENCH_LABEL`和`KARESERVE_RESULT_FILENAME`控制结果文件名。

`KARESERVE_BENCH_DRY_RUN=1`只输出最终 `vllm bench serve`命令，用于检查模型、数据集和结果参数。
