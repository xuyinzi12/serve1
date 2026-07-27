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
- `windowed_prefix`：按共享 Prefix 形成逻辑请求组，并联合考虑 GPU 本地缓存命中、实例负载和 KVCache 容量压力。组内请求仍以独立 HTTP 请求转发，vLLM 负责实际执行批次。

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
    "kv_cache_weight": 2.0,
    "kv_cache_high_watermark": 0.8,
    "kv_cache_hard_limit": 0.95,
    "decode_token_weight": 4.0,
    "group_block_size": 16
  }
}
```

`windowed_prefix` 将 GPU 本地命中 token 换算为复用收益，将运行请求、等待请求和窗口内新增工作换算为负载代价。窗口内新增工作使用目标实例未命中的 Prompt token 和加权 Decode token 计算。`kv_cache_high_watermark` 以上的容量占用产生二次增长惩罚；存在低于 `kv_cache_hard_limit` 的实例时，高于该阈值的实例退出候选集；全部实例超过阈值时，Router 选择占用最低的实例。vLLM `/metrics` 拉取失败的实例在存在健康实例时退出候选集。

所有实例连接同一个 LMCache MP Server 时，LMCache CPU 命中对实例排序构成共享条件。Router 使用 GPU 本地 Prefix 和实例负载完成选点，目标 vLLM Connector 执行 LMCache Lookup 和 Load。

## 硬件测速

CPU 到 GPU 的 KVCache 加载路径使用 pinned host-to-device 测速。脚本通过 CUDA Event 测量多个数据尺寸，并输出线性拟合带宽、固定延迟和各尺寸分位延迟：

```bash
.venv-vllm-0.26-lmcache/bin/python \
  scripts/benchmark/measure_h2d_bandwidth.py \
  --device 1 \
  --sizes-mib 1,4,16,64 \
  --warmup 10 \
  --iterations 30
```

测速结果写入运行配置的 `hardware_profile`：

```json
{
  "hardware_profile": {
    "h2d_bandwidth_gbps": 0.0,
    "h2d_base_latency_ms": 0.0,
    "measurement": "scripts/benchmark/measure_h2d_bandwidth.py"
  }
}
```

当前路由策略输出该 Profile用于实验记录。共享 LMCache场景中的实例排序暂不使用 H2D参数。后续加入外部缓存请求级目录后，Router使用该 Profile估算 LMCache加载时间。

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

LMCache Server 默认监听 `127.0.0.1:5555`，CPU L1 缓存容量为 2 GiB，淘汰策略为 LRU。LMCache 自身 observability 在调试配置中关闭，vLLM external prefix cache metrics 保持启用。vLLM 使用 `LMCacheMPConnector`、`kv_both` 角色和非 Hybrid KV Cache Manager。单 GPU 测试通过保持 LMCache Server 运行并顺序重启 vLLM，验证外部 KVCache 的 Store、Lookup 和 Retrieve。`scripts/debug/lmcache_probe.py` 提供确定性长 Prefix 请求和缓存指标输出。KaReserve 从各 vLLM `/metrics` 读取 LMCache 查询量和命中量，并通过 `/routing/state` 输出累计命中率。跨 GPU 并发共享需要两张空闲 GPU。

## 项目边界

KaReserve 管理请求窗口、Prefix逻辑分组、GPU本地 Prefix感知、容量与负载感知选点和 HTTP转发。vLLM 管理执行批次和 GPU KVCache。LMCache通过 vLLM KV Connector管理共享 CPU KVCache。当前单机路由不执行 GPU间 KVCache传输，也不在转发前调用 LMCache Lookup。vLLM连接器负责实际请求的 LMCache命中查询、加载和缓存锁生命周期。
