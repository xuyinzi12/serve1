# KaReserve

KaReserve 是位于客户端与多个 vLLM 实例之间的 KVCache 感知路由器。Router 在每个请求到达时查询 GPU Prefix Cache、可选的 LMCache 缓存目录和实例负载，再把请求转发到预计完成时间最低的实例。vLLM 独立执行 Continuous Batching 和模型推理。

```text
Client
  → OpenAI HTTP
  → KaReserve Router
     Tokenizer → Cache State → Route Planner → vLLM instance
                         ↘ LMCache memory / disk
```

## 目录

```text
kareserve/
  server.py           HTTP 接口和组件生命周期
  routing.py          单请求候选构造、选点和资源预留
  policy.py           路由基线与分层完成时间策略
  performance.py      Prefill 与在线排队时间模型
  tracker.py          GPU 缓存目录和实例运行状态
  lmcache_client.py   Router 到 LMCache 的只读目录查询
  lmcache_server.py   LMCache 目录查询扩展
  tokenizer.py        Router 本地分词
  state.py            共享状态类型
configs/config.json   部署配置
scripts/              启动、停止、测量和 Benchmark 入口
docs/architecture.md  路由状态与代价模型
```

## 路由策略

`round_robin`提供无缓存感知基线；`gpu_prefix_load`使用 GPU Prefix 命中、实例负载和容量；`tiered_completion_time`增加 LMCache 主机内存、文件系统或对象存储的访问成本。三种策略通过`KARESERVE_POLICY_OVERRIDE`选择。

主策略计算：

```text
预计完成时间 = Prompt 路径时间 + 排队时间 + 容量压力
```

Prompt 路径包含缓存加载和剩余 Prefill。排队时间由 vLLM 队列指标与请求完成记录共同估计。Router 不使用聚合窗口，也不控制 vLLM 的执行批次。

## 配置

`configs/config.json`描述 vLLM 节点、LMCache 缓存域、Tokenizer、路由参数和硬件性能档案。增加实例时只需要向`nodes`添加节点。共享同一个 LMCache 服务的实例使用相同的`cache_domain_id`。

`hardware_profile.prefill_time_model`保存当前模型与 GPU 组合的 Prefill 测量结果；`host_memory_to_gpu_bandwidth_gbps`表示主机运行内存到 GPU 显存的 KVCache 有效传输带宽；`medium_profiles`描述文件系统或对象存储路径。完整的介质成本比较需要这些数值使用同一台部署机器的实测结果。

## 启动

服务器环境位于`/home/zn/xyz/serve1/.venv-vllm-0.26`。以下命令使用配置中的节点端口启动 vLLM 和 Router：

```bash
cd /home/zn/xyz/serve1
GPU_IDS=0,3 bash scripts/start.sh
curl http://127.0.0.1:8090/health
curl http://127.0.0.1:8090/routing/state
```

默认关闭 LMCache。分层缓存实验使用：

```bash
KARESERVE_ENABLE_LMCACHE=1 GPU_IDS=0,3 bash scripts/start.sh
```

基础 Prefix 工作负载使用 vLLM Benchmark：

```bash
bash scripts/run_benchmark.sh
```

停止本项目启动的进程：

```bash
bash scripts/stop.sh
```

## 数据集

服务器现有 ShareGPT 数据位于`/home/zn/datasets/sharegpt/shareGPT/computer_zh_26k.jsonl`。转换器只读取源文件，并把兼容 vLLM Benchmark 的结果写入`runtime/`：

```bash
.venv-vllm-0.26/bin/python scripts/convert_sharegpt.py \
  --input /home/zn/datasets/sharegpt/shareGPT/computer_zh_26k.jsonl \
  --output runtime/datasets/sharegpt-zh.json
```

`runtime/`保存当前机器生成的日志、PID、LMCache 数据和实验结果，该目录不进入 Git 仓库。

## 接口与观测

Router 透传`POST /v1/completions`和`POST /v1/chat/completions`，提供`POST /tokenize`、`POST /detokenize`、`GET /health`和`GET /routing/state`。响应头包含目标实例、分介质 Prefix 命中、预计成本和路由规划时间。

结构化日志使用相同的`request_id`关联`route_decision`与`route_result`。选点记录包含缓存路径、预测成本和规划耗时；执行记录包含首个有效输出时间、完整响应时间和完成状态。在线排队模型的样本数、斜率和误差位于`/routing/state`的`route_planner.queue_estimator`。
