# KaReserve启动测试教程

## 1. 进入项目并检查环境

```bash
cd /home/zn/xyz/serve1

.venv-vllm-0.26/bin/python -c \
  "import vllm, torch, lmcache; print(vllm.__version__, torch.__version__, lmcache.__version__)"

nvidia-smi
```

预期版本为vLLM 0.26.0、Torch 2.11.0+cu130和LMCache 0.5.2。项目只使用`.venv-vllm-0.26`。

## 2. 理解端口

单个vLLM实例开放三个项目端口：

```text
GPU0  HTTP 8101  KV Event 5557  Replay 6557
GPU1  HTTP 8102  KV Event 5558  Replay 6558
GPU2  HTTP 8103  KV Event 5559  Replay 6559
```

HTTP端口接收推理请求并提供`/metrics`。KV Event端口通过ZMQ发布GPU缓存块的Store、Remove和Clear事件。Replay端口补发发布端仍保留的事件。LMCache使用ZMQ 5555提供KV操作，并使用HTTP 8080提供管理与健康检查。Router使用HTTP 8090接收Benchmark请求，并在本地执行Tokenization。

## 3. 执行单GPU连通性测试

该测试使用vLLM内置Prefix Repetition生成器，不需要磁盘数据集：

```bash
.venv-vllm-0.26/bin/python \
  scripts/experiment/run_experiment.py \
  --manifest configs/experiments/smoke.json
```

运行器会停止本项目旧进程、启动8 GiB LMCache、启动GPU1上的vLLM、等待服务健康、启动Router、发送32个请求、保存结果并停止本轮进程。

Benchmark默认使用`temperature=0`和`ignore_eos=true`，相同输入产生确定性采样路径并执行固定输出长度。

结果位于：

```text
runtime/experiments/smoke/run-01/result.json
runtime/experiments/smoke/run-01/run-manifest.json
runtime/experiments/smoke/run-01/router-state.json
runtime/experiments/smoke/summary.json
```

## 4. 生成确定性Prefix Trace

双GPU路由实验使用可重复回放的Timed Trace：

```bash
.venv-vllm-0.26/bin/python \
  scripts/data/build_prefix_trace.py \
  --output runtime/datasets/prefix-trace.jsonl \
  --num-requests 256 \
  --num-prefixes 8 \
  --prefix-len 512 \
  --suffix-len 64 \
  --output-len 16 \
  --request-rate 100 \
  --seed 0
```

每行记录请求时间、输入长度、输出长度和Prefix Chunk标识。具有相同`prefix_id`的请求共享前512个Token。`PYTHONHASHSEED=0`固定vLLM从Chunk标识生成的Token。

## 5. 执行双GPU Prefix路由实验

```bash
.venv-vllm-0.26/bin/python \
  scripts/experiment/run_experiment.py \
  --manifest configs/experiments/prefix-routing.json
```

该Manifest使用GPU0和GPU1、32 GiB LMCache、2 ms窗口、256条Timed Trace请求和3次独立冷启动重复。启动前需要确认GPU0和GPU1可用。

## 6. 转换服务器ShareGPT

协作者ShareGPT位置为：

```text
/home/zn/datasets/sharegpt/shareGPT/computer_en_26k.jsonl
/home/zn/datasets/sharegpt/shareGPT/computer_zh_26k.jsonl
/home/zn/datasets/sharegpt/sharegpt_jsonl/
```

源数据每个回合包含`human`和`assistant`。第一轮请求只发送`human`。多轮累计模式把历史`human`和历史`assistant`作为后续Prompt上下文，当前待生成的assistant内容不会发给模型。Benchmark使用`output_tokens`控制本轮模型生成长度。

```bash
.venv-vllm-0.26/bin/python \
  scripts/data/convert_sharegpt.py \
  --input /home/zn/datasets/sharegpt/shareGPT/computer_en_26k.jsonl \
  --output runtime/datasets/sharegpt-custom.jsonl \
  --limit 256 \
  --output-tokens 64 \
  --mode cumulative
```

转换结果使用vLLM `custom`格式：

```json
{"prompt": "Human: ...\nAssistant:", "output_tokens": 64}
```

执行自然请求实验：

```bash
.venv-vllm-0.26/bin/python \
  scripts/experiment/run_experiment.py \
  --manifest configs/experiments/sharegpt-routing.json
```

## 7. 执行策略消融

复制`configs/experiments/prefix-routing.json`并修改`name`和`router_overrides.policy`：

```text
round_robin
least_load
prefix_hash
windowed_prefix
```

窗口消融使用`window_ms=0`和`window_ms=2`。LMCache消融使用`runtime.enable_lmcache=false`和`true`。每个Manifest保持相同数据集、seed、模型、GPU、请求率和重复次数。

## 8. 验证LMCache跨vLLM进程复用

单实例持续运行时，vLLM优先使用本实例GPU Prefix Cache。LMCache持久性测试保持LMCache Server运行，发送一次长Prefix请求，重启vLLM进程，随后发送相同请求：

```bash
GPU_IDS=1 bash scripts/experiment/verify_lmcache_persistence.sh
```

第一阶段负责Store，第二阶段负责Retrieve。第二阶段的`external_prefix_cache_hits_total`增长构成CPU KVCache复用证据。

LMCache MP Connector当前不发布CPU L1块级Store和Evict事件。Router在请求成功完成后登记完整外部缓存Chunk，并把该目录用于路由成本估算。`/routing/state`中的`monitoring`保存vLLM实际Lookup与命中累计值，`cache_catalog.external_source`标明预测目录来源。vLLM Connector对每个请求执行最终Lookup。

## 9. 手工调试

手工启动单GPU Stack：

```bash
GPU_IDS=1 \
KARESERVE_CONFIG_PATH=/home/zn/xyz/serve1/configs/router.single-node.json \
LMCACHE_L1_SIZE_GB=32 \
bash scripts/debug/start_stack.sh
```

检查状态：

```bash
curl http://127.0.0.1:8080/healthcheck
curl http://127.0.0.1:8102/health
curl http://127.0.0.1:8090/health
curl http://127.0.0.1:8090/routing/state
```

查看日志：

```bash
tail -f runtime/logs/lmcache-server.log
tail -f runtime/logs/vllm-gpu1.log
tail -f runtime/logs/kareserve.log
```

停止项目进程：

```bash
bash scripts/debug/stop_debug_cluster.sh
```

停止脚本只处理`runtime/pids`记录且命令路径属于当前项目的进程。
