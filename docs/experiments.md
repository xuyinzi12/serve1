# 实验说明

## 配置关系

一次正式实验由一个Manifest描述。`runtime`指定GPU、模型和LMCache容量；`router_config`指定节点、缓存域、策略参数和硬件Profile；`benchmark`指定数据集、请求数量、到达率和并发；`router_overrides`用于策略与窗口消融。`scripts/experiment/run_experiment.py`把Manifest转换为各进程需要的环境变量，并保存解析后的完整运行记录。

调试脚本中的环境变量只覆盖单轮运行参数。固定拓扑和算法参数保留在Router JSON中，正式实验参数保留在Manifest中。

## 数据集

`prefix_repetition`由vLLM Benchmark直接生成，可控制公共Prefix长度、Suffix长度、Prefix数量和输出长度，适合验证路由机制。当前Smoke Manifest使用该数据集。

外部数据通过`dataset_name`和`dataset_path`进入vLLM Benchmark。项目提供ShareGPT转换器和确定性Trace生成器：

```bash
.venv-vllm-0.26/bin/python scripts/data/convert_sharegpt.py --help
.venv-vllm-0.26/bin/python scripts/data/build_prefix_trace.py --help
```

服务器现有原始数据位于`/home/zn/datasets/`，其中包含ShareGPT、LongBench和ArXiv Summarization。转换后的实验输入写入`runtime/datasets/`，该目录不进入Git。

## 基线

正式对比保持模型、数据集、请求轨迹、请求率、并发、seed和缓存初始状态一致。

| 策略 | 决策依据 |
|---|---|
| `round_robin` | 到达顺序轮询 |
| `least_load` | vLLM队列与Router在途工作量 |
| `prefix_hash` | 固定长度Prefix哈希映射 |
| `windowed_prefix`, `window_ms=0` | 单请求缓存与负载感知 |
| `windowed_prefix`, `window_ms>0` | 窗口级Prefix、介质、负载与容量联合选点 |

LMCache消融使用同一策略分别设置`runtime.enable_lmcache=false`和`true`。`runtime.reset_lmcache=true`会在每轮启动前清空项目`runtime`目录内的L2数据；L1和GPU缓存随进程停止释放。该流程保证各轮使用一致的冷缓存状态。

## 执行

Smoke Test验证启动、请求转发、缓存查询和结果保存：

```bash
cd /home/zn/xyz/serve1
.venv-vllm-0.26/bin/python scripts/experiment/run_experiment.py \
  --manifest configs/experiments/smoke.json
```

Prefix路由实验使用确定性Trace：

```bash
.venv-vllm-0.26/bin/python scripts/experiment/run_experiment.py \
  --manifest configs/experiments/prefix-routing.json
```

ShareGPT实验使用转换后的请求文件：

```bash
.venv-vllm-0.26/bin/python scripts/experiment/run_experiment.py \
  --manifest configs/experiments/sharegpt-routing.json
```

`--dry-run`只校验Manifest并输出执行链路，`--leave-running`在最后一轮结束后保留服务。

## 结果

每轮结果位于`runtime/experiments/<name>/run-<index>/`。`run-manifest.json`记录原始Manifest、解析后的环境变量、Git提交和依赖版本；`result.json`保存vLLM Benchmark指标；`router-state.json`保存实验结束时的Router与缓存状态。所有轮次完成后，`summary.json`汇总成功率、吞吐量、TTFT和端到端延迟。

Router日志中的`route_decision`记录节点选择、GPU/主机内存/磁盘Prefix长度、窗口等待和预计成本。响应头提供同一组请求级信息，便于对照Benchmark结果。

## 专项验证

LMCache跨vLLM进程持久性使用以下脚本：

```bash
GPU_IDS=1 bash scripts/experiment/verify_lmcache_persistence.sh
```

硬件Profile使用项目测速脚本生成。H2D测速写入CPU介质参数，文件系统读取测速写入FS介质参数：

```bash
.venv-vllm-0.26/bin/python scripts/benchmark/measure_h2d_bandwidth.py --help
.venv-vllm-0.26/bin/python scripts/benchmark/measure_storage_bandwidth.py --help
```
