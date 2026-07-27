# KaReserve

KaReserve 是部署在多个 vLLM 实例前方的 Prefix-aware Router。Router 在短时间窗口内收集请求，根据公共 Token Prefix、各实例的本地 KVCache 和运行负载完成实例分配。vLLM 继续负责 Continuous Batching、Paged KVCache 和模型执行。

## 请求链路

```text
Benchmark / Client
        ↓
KaReserve Router
  1. 调用 vLLM /tokenize
  2. 进入短时 Request Pool
  3. 执行窗口级路由策略
  4. 转发到目标 vLLM
        ↓
vLLM Continuous Batching
```

Router 提供以下接口：

- `POST /tokenize`
- `POST /v1/completions`
- `POST /v1/chat/completions`
- `GET /health`
- `GET /routing/state`

Completion 响应包含目标实例、路由策略、窗口大小、Router 等待时间和 Prefix 命中长度等响应头。Router 日志为每个请求记录一条 `route_decision` JSON。

## 路由策略

`routing.policy` 支持以下值：

- `round_robin`：按照窗口到达顺序轮询实例。
- `least_load`：按照实例负载和窗口内虚拟负载分配。
- `prefix_hash`：将固定长度 Prefix 稳定映射到实例。
- `windowed_prefix`：按共享 Prefix 分组，并联合考虑缓存命中与实例负载。

`routing.window_ms=0` 表示逐请求立即分配。正数窗口表示 Router 最多等待对应时间收集请求。`routing.max_batch_size` 会提前触发窗口处理。

```json
{
  "routing": {
    "policy": "windowed_prefix",
    "window_ms": 2.0,
    "max_batch_size": 64,
    "prefix_hash_tokens": 256,
    "prefix_tokens_per_load_unit": 256.0,
    "queue_weight": 1.0,
    "group_block_size": 16
  }
}
```

## 运行环境

服务器独立环境位于：

```text
/home/zn/xyz/serve1/.venv-vllm-0.26
```

该环境与 `/home/zn/vllm_advanced_env` 隔离。源码通过工作目录直接加载，当前调试流程不要求安装 KaReserve wheel。

## 调试启动

调试脚本只支持 GPU 0、GPU 1、GPU 2。每张 GPU 对应独立的 HTTP 端口和 KV Event 端口。

```bash
cd /home/zn/xyz/serve1
GPU_IDS=1 bash scripts/debug/start_vllm_cluster.sh

KARESERVE_CONFIG_PATH=/home/zn/xyz/serve1/examples/config.single-node.json \
  bash scripts/debug/start_router.sh
```

双实例启动需要确认两张 GPU 均为空闲：

```bash
GPU_IDS="0 1" bash scripts/debug/start_vllm_cluster.sh
```

停止项目启动的全部调试进程：

```bash
bash scripts/debug/stop_debug_cluster.sh
```

停止脚本根据 PID 文件和 serve1 命令路径核对进程归属。运行日志、PID、Benchmark 结果和独立环境均位于 Git 忽略目录。

## vLLM Benchmark

Router 接口兼容 `vllm bench serve` 的 Chat Completions 和 Completions 请求。Prefix 基础测试示例：

```bash
.venv-vllm-0.26/bin/vllm bench serve \
  --backend openai-chat \
  --base-url http://127.0.0.1:8090 \
  --endpoint /v1/chat/completions \
  --model kareserve-opt-1.3b \
  --tokenizer /home/zn/llm_models/opt-1.3b \
  --dataset-name prefix_repetition \
  --num-prompts 32 \
  --prefix-repetition-prefix-len 512 \
  --prefix-repetition-suffix-len 64 \
  --prefix-repetition-num-prefixes 4 \
  --prefix-repetition-output-len 16 \
  --request-rate 20
```

正式策略对比需要复用相同模型、seed、请求轨迹和冷缓存启动流程。Round Robin、Prefix Hash、Window=0 和 Windowed Prefix 分别运行并独立保存结果。

## LMCache MP

LMCache 使用独立环境：

```text
/home/zn/xyz/serve1/.venv-vllm-0.26-lmcache
```

该环境通过 `.pth` 只读引用基础 vLLM 0.26 环境，并单独安装 LMCache 0.5.2、SortedContainers 和 CuPy CUDA 13。基础环境中的 vLLM、Torch、CUDA、OpenTelemetry 和 Prometheus 版本不会发生变化。

```bash
bash scripts/env/install_lmcache_overlay.sh
```

启动独立 LMCache Server：

```bash
bash scripts/debug/start_lmcache_server.sh
```

启动连接 LMCache MP Server 的 vLLM：

```bash
VLLM_ENV=/home/zn/xyz/serve1/.venv-vllm-0.26-lmcache \
KARESERVE_LMCACHE_MP=1 \
GPU_IDS=1 \
bash scripts/debug/start_vllm_cluster.sh
```

LMCache Server 默认监听 `127.0.0.1:5555`，CPU L1 缓存容量为 2 GiB，淘汰策略为 LRU。LMCache 自身 observability 在调试配置中关闭，vLLM external prefix cache metrics 保持启用。vLLM 使用 `LMCacheMPConnector`、`kv_both` 角色和非 Hybrid KV Cache Manager。单 GPU 测试通过保持 LMCache Server 运行并顺序重启 vLLM，验证外部 KVCache 的 Store、Lookup 和 Retrieve。`scripts/debug/lmcache_probe.py` 提供确定性长 Prefix 请求和缓存指标输出。跨 GPU 并发共享需要两张空闲 GPU。

## 项目边界

KaReserve 管理请求窗口、Prefix 分组、实例选择和 HTTP 转发。vLLM 管理执行批次和 GPU KVCache。LMCache 通过 vLLM KV Connector 管理外部 KVCache，LMCache MP Server 的跨实例共享需要独立安装与验证。
