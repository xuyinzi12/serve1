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

`configs/config.json` 是唯一配置文件。`nodes` 描述当前运行的 vLLM 实例；单实例配置包含一个节点，增加实例时向同一数组增加节点。`cache_domains` 描述各节点使用的 LMCache 服务；同一台主机内共享一个 LMCache 的实例使用相同的 `cache_domain_id`。缓存域的`model_id`使用vLLM加载模型时的模型路径，该值对应LMCache布局注册键。

`hardware_profile.host_memory_to_gpu_bandwidth_gbps` 表示 KVCache 从主机运行内存复制到 GPU 显存的有效带宽。`prefill_time_model`保存当前模型与GPU组合的离线二阶回归结果，模型输入为请求长度和GPU Prefix命中长度，输出为预计Prefill TTFT。主机内存与文件系统传输使用KVCache字节数、实测带宽和固定延迟计算毫秒成本。

`routing.inflight_prefix_reuse_probability`表示Router对执行中Prefix可被同窗口后续请求复用的估计概率。当前vLLM没有提供该执行协议，默认值保持为`0.0`。Policy仍执行窗口级联合分配，并在共置造成排队或容量压力时拆分Prefix组。

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

Prefill模型使用空闲vLLM实例离线标定：

```bash
.venv-vllm-0.26/bin/python scripts/profile_prefill.py --endpoint http://127.0.0.1:8101 --model kareserve-opt-1.3b --tokenizer /home/zn/llm_models/opt-1.3b --output runtime/prefill-profile.json
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

Router 透传`POST /v1/completions`和`POST /v1/chat/completions`，并提供`POST /tokenize`、`POST /detokenize`、`GET /health`和`GET /routing/state`。Tokenizer接口由Router本地执行，Benchmark和Prefix路由共用同一Token序列。当前 OPT 基础联调使用 Completion 请求。Chat 请求需要模型 Tokenizer 自带 Chat Template，或通过`KARESERVE_CHAT_TEMPLATE`为 Router 与 vLLM 同时指定同一模板。

Router 负责请求聚合、Prefix 分组、缓存状态查询、实例选择和 HTTP 转发。vLLM 负责实际执行批次。LMCache Connector 负责主机内存或磁盘中的 KVCache 加载。

`runtime/`只保存当前机器生成的日志、PID、LMCache数据和Benchmark结果。该目录由脚本按需创建，不进入Git仓库。

Router为每个请求记录`route_decision`和`route_result`两类结构化日志。前者保存选点时的Prefix命中、聚合等待和预测成本；后者保存上游连接、首个有效输出事件、完整响应耗时和响应字节数。两类记录通过`request_id`关联。`estimated_cost_unit=ms`表示当前配置已加载Prefill时间模型，`normalized`表示策略仍在使用归一化工作量。
