# KaReserve

KaReserve 是部署在多个 vLLM 实例前方的 Prefix-aware Router。Router 在短时间窗口内收集请求，查询各实例的 GPU Prefix Cache、共享 LMCache 和运行负载，然后完成实例分配。vLLM 继续管理 Continuous Batching、Paged KVCache 和模型执行。

```text
Benchmark / Client
        │ OpenAI HTTP
        ▼
KaReserve Router
  Tokenizer → Request Pool → Cache Lookup → Policy
        │
        ├── vLLM instance 0 ── GPU KVCache
        └── vLLM instance 1 ── GPU KVCache
                    │
                    └── LMCache ── host memory / filesystem
```

## 目录

```text
kareserve/            Router实现
configs/router.*      节点、缓存域和策略配置
configs/experiments/  可复现实验Manifest
scripts/debug/        服务启动与停止
scripts/benchmark/    vLLM Benchmark、测速和结果汇总
scripts/data/         数据转换与Trace生成
scripts/experiment/   实验运行与校验
docs/architecture.md  系统状态与决策逻辑
docs/experiments.md   数据集、基线和执行流程
```

## 启动

服务器运行环境位于`/home/zn/xyz/serve1/.venv-vllm-0.26`，其中包含 vLLM 0.26.0、LMCache 0.5.2、Torch 和 CUDA。脚本直接使用该环境，无需执行`activate`。

单实例调试使用 GPU 1：

```bash
cd /home/zn/xyz/serve1
GPU_IDS=1 bash scripts/debug/start_stack.sh
curl http://127.0.0.1:8090/routing/state
```

停止本项目启动的进程：

```bash
bash scripts/debug/stop_debug_cluster.sh
```

双实例运行使用与 GPU 端口映射一致的配置：

```bash
KARESERVE_CONFIG_PATH=/home/zn/xyz/serve1/configs/router.two-node.json \
GPU_IDS="0 1" \
bash scripts/debug/start_stack.sh
```

## 实验入口

Manifest 同时固定模型、GPU、LMCache、Router配置、数据集和请求负载。Smoke Test 的执行命令如下：

```bash
.venv-vllm-0.26/bin/python scripts/experiment/run_experiment.py \
  --manifest configs/experiments/smoke.json
```

运行记录写入`runtime/experiments/<name>/`，其中包含解析后的Manifest、软件版本、Git提交、Benchmark结果、Router状态和汇总结果。详细的工作负载与对照组见[实验说明](docs/experiments.md)。

## 接口

Router透传`POST /v1/completions`和`POST /v1/chat/completions`，并提供`GET /health`和`GET /routing/state`。响应头记录目标实例、窗口大小、Router等待时间、各介质Prefix命中长度和预计代价。服务日志为每个请求写入一条`route_decision` JSON。

## 当前边界

Router负责请求聚合、Prefix分组、GPU与LMCache状态查询、实例选择和HTTP转发。vLLM负责实际执行批次。LMCache Connector负责缓存锁、主机内存或磁盘数据加载以及GPU Block写入。当前实现不执行GPU间KVCache迁移，也不合并首次并发Prefix Miss。完整的数据来源和成本模型见[架构说明](docs/architecture.md)。
