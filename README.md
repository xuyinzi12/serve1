# KaReserve

KaReserve 是部署在多个 vLLM 服务实例之前的窗口化 Prefix-aware Router。Router 在有限时间窗口内收集请求，按照公共 token Prefix 建立逻辑请求组，并结合各实例的 KVCache 状态与运行负载完成请求分配。

## 系统边界

KaReserve 调用 vLLM `/tokenize` 获取应用 Chat Template 后的 token IDs，调用 `/metrics` 获取实例负载，通过 ZMQ 订阅 vLLM KVCache events，通过 `/v1/chat/completions` 转发推理请求。vLLM Scheduler 管理 Continuous Batching 和 GPU KV blocks。LMCache 通过 vLLM KVConnector 管理外部 KVCache lookup 与 load。

## 安装

```bash
git clone <repository-url>
cd kareserve
python -m venv .venv
source .venv/bin/activate
pip install .
```

## 配置

```bash
cp config.example.json config.json
```

`group_block_size` 需要与 vLLM 的 Prefix Hash 粒度保持一致。每个 vLLM 实例需要暴露 OpenAI API、`/tokenize`、`/metrics` 和独立的 KV event endpoint。

vLLM 需要启用 Prefix Caching 和 KV event 发布。以下参数展示单节点事件配置：

```bash
vllm serve <model> \
  --enable-prefix-caching \
  --kv-events-config \
  '{"enable_kv_cache_events":true,"publisher":"zmq","endpoint":"tcp://*:5557"}'
```

不同物理服务器可以使用相同监听端口。同一服务器上的多个 vLLM 实例需要使用不同端口。LMCache 按现有 vLLM KVConnector 部署方式配置。

## 启动

```bash
kareserve-server --config config.json --host 0.0.0.0 --port 8080
```

客户端将 OpenAI Chat Completions 请求发送到：

```text
http://<router-host>:8080/v1/chat/completions
```

健康检查地址为：

```text
http://<router-host>:8080/health
```

## 当前调度流程

```text
vLLM Tokenization
→ 窗口请求聚合
→ 公共 Prefix 分组
→ KVCache 与实例负载查询
→ 窗口内请求组分配
→ 请求级 HTTP 转发
→ vLLM Continuous Batching
```

当前版本实现 Prefill vLLM 实例选择。当前版本不包含 Prefill/Decode Pair 选择、Router 主动 KVCache 传输和 vLLM 内部执行 Batch 控制。
