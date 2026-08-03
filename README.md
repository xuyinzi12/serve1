# KaReserve

KaReserve 位于客户端与 vLLM 实例之间。Router 在短时间窗口内收集请求，查询 GPU Prefix Cache、LMCache 和实例负载，再把每个请求转发到选定实例。vLLM 负责实例内部的 Continuous Batching 和模型执行。

```text
Client
  │ OpenAI HTTP
  ▼
KaReserve Router
  Tokenizer → Request Pool → Cache State → Routing Policy
                                           │
                                           ▼
                                      vLLM instance
                                      GPU KVCache
                                           │
                                           ▼
                                      LMCache memory / disk
```

## 目录

```text
kareserve/
  server.py            HTTP入口和组件生命周期
  request_pool.py      请求窗口与批量路由
  policy.py            节点选择和成本模型
  tracker.py           GPU缓存目录和实例指标
  lmcache_client.py    Router侧LMCache查询
  lmcache_server.py    LMCache目录查询服务
  tokenizer.py         Router本地分词
  state.py             共享状态类型
configs/config.json    当前部署配置
scripts/               启动、停止、验证和测速入口
docs/architecture.md   状态来源与路由逻辑
```

## 配置

`configs/config.json` 是唯一配置文件。`nodes` 描述当前运行的 vLLM 实例；单实例配置包含一个节点，增加实例时向同一数组增加节点。`cache_domains` 描述各节点使用的 LMCache 服务；同一台主机内共享一个 LMCache 的实例使用相同的 `cache_domain_id`。

`hardware_profile.host_memory_to_gpu_bandwidth_gbps` 表示 KVCache 从主机运行内存复制到 GPU 显存的有效带宽。`prefill_ms_per_token` 表示当前模型在目标 GPU 上执行 Prefill 的平均每 Token 时间。当前值为空，Policy统一使用归一化工作量，Router输出的代价不解释为毫秒。后续取得Prefill实测值后，Policy才启用基于时间的介质成本。

## 基础联调

服务器环境位于`/home/zn/xyz/serve1/.venv-vllm-0.26`。默认配置对应 GPU1、vLLM 端口 8102、LMCache HTTP 端口 8080 和 Router 端口 8090。

```bash
cd /home/zn/xyz/serve1
GPU_IDS=1 bash scripts/start.sh
curl http://127.0.0.1:8090/health
curl http://127.0.0.1:8090/routing/state
```

基础 Prefix 请求使用 vLLM 自带的 Benchmark 生成器：

```bash
bash scripts/run_benchmark.sh
```

LMCache 跨 vLLM 进程恢复检查使用：

```bash
GPU_IDS=1 bash scripts/verify_lmcache.sh
```

停止本项目启动的进程：

```bash
bash scripts/stop.sh
```

## 接口

Router 透传`POST /v1/completions`和`POST /v1/chat/completions`，并提供`GET /health`和`GET /routing/state`。当前 OPT 基础联调使用 Completion 请求。Chat 请求需要模型 Tokenizer 自带 Chat Template，或通过`KARESERVE_CHAT_TEMPLATE`为 Router 与 vLLM 同时指定同一模板。

Router 负责请求聚合、Prefix 分组、缓存状态查询、实例选择和 HTTP 转发。vLLM 负责实际执行批次。LMCache Connector 负责主机内存或磁盘中的 KVCache 加载。

`runtime/`只保存当前机器生成的日志、PID、LMCache数据和Benchmark结果。该目录由脚本按需创建，不进入Git仓库。
