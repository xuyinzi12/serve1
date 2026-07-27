# KaReserve 架构

## 当前系统

KaReserve 运行在多个完整 vLLM 实例前方。每个 vLLM 实例同时执行 Prefill 和 Decode，并管理本实例 GPU KVCache。单机 LMCache MP Server提供共享 CPU KVCache。Router 管理请求聚合、Prefix逻辑分组、实例状态采集、窗口级选点和 HTTP转发。

```text
Benchmark / Client
        ↓
KaReserve Router
  Request Pool
  Prefix Grouper
  KVCache Tracker
  Windowed Policy
        ↓
完整 vLLM实例
  Continuous Batching
  GPU KVCache
  LMCache Connector
        ↓
单机 LMCache MP Server
  CPU KVCache
```

Router 使用 vLLM `/tokenize`获得 Token IDs，使用 KV Event维护实例 GPU Prefix目录，使用 `/metrics`维护运行请求、等待请求、KVCache占用和外部缓存累计命中状态。Router 的 `windowed_prefix` 策略使用 GPU本地 Prefix复用收益、实例负载、窗口虚拟工作量和 KVCache容量压力完成联合选点。

Prefix组表示窗口内具有公共 Token Prefix的一组请求。组内请求获得同一个目标实例，Router仍然逐请求转发，vLLM决定实际执行批次。当前实现不提供首次并发 Miss合并，也不提供 GPU间 KVCache传输。

LMCache Lookup与Load发生在请求进入目标 vLLM之后。所有本机 vLLM连接同一个 LMCache MP Server时，CPU缓存命中构成实例间共享条件。Router当前不使用 LMCache命中状态区分本机实例。

## 多实体架构

多实体架构把每台物理主机视为一个独立缓存与执行实体。每个实体拥有独立 GPU、CPU内存和本地磁盘：

```text
实体 A
  vLLM A
  GPU KVCache A
  LMCache A / CPU内存 A
  Disk Cache A

实体 B
  vLLM B
  GPU KVCache B
  LMCache B / CPU内存 B
  Disk Cache B

Global Router
  实体状态
  全局 Prefix目录
  路由代价模型
```

多实体路由需要判断请求 Prefix位于哪个实体、位于该实体的哪个介质、目标实例需要等待多久。第一版多实体策略优先把请求路由到持有缓存的实体，避免跨实体搬运 KVCache。远端缓存传输作为后续独立能力。

### 全局 Prefix目录

每个实体在 KVCache Store、Evict 和 Clear时发布元数据事件。全局目录维护 `Prefix Chunk → 实体 → 介质 → 状态版本`映射。Router使用请求 Token IDs计算 Chunk Hash，并查询最长连续命中长度。

目录事件需要包含模型标识、Chunk Hash、Chunk Size、缓存介质、实体标识、对象大小、事件版本和时间戳。目录允许短暂陈旧，目标 vLLM与本地 LMCache执行最终命中校验。事件版本、心跳和 TTL负责清理失联实体与过期目录项。

### 分层代价模型

每个实体维护独立 Hardware Profile：

```text
GPU本地命中代价
CPU内存到GPU带宽与固定延迟
磁盘到CPU带宽与固定延迟
节点间网络带宽与固定延迟
vLLM排队状态
GPU、CPU和磁盘容量压力
```

Router 对请求和实体计算：

```text
预计完成代价
= 实例排队时间
+ 命中介质加载时间
+ 未命中 Prefix计算时间
+ 容量压力惩罚
```

共享 Prefix组使用组内总命中收益与目标实体新增工作量完成联合选点。所有实体均为冷 Miss时，策略按照负载和容量分散请求。该启发式策略构成多实体基线，后续优化器使用同一状态与代价接口。

### 每实体 LMCache

每台主机运行独立 LMCache服务。CPU内存构成本机快速外部缓存，本地磁盘构成本机容量层。Router需要获得各实体外部缓存的只读目录状态。实现路径包含 LMCache Controller集成或 Store、Evict、Clear事件驱动的 KaReserve目录。

目标请求到达实体后，本地 vLLM Connector负责 Lookup、缓存锁、CPU或磁盘加载和GPU Block写入。Router负责选择实体与估算代价，执行层负责最终数据正确性。

### 状态与故障处理

实体状态需要包含 HTTP健康、指标更新时间、运行队列、GPU KVCache占用、CPU缓存占用、磁盘缓存占用和传输队列。Router对失联实体停止新分配，并保留短期状态用于恢复。请求转发失败需要记录失败阶段；尚未进入模型执行的请求可以选择其他实体重新分配。

### 验证路径

多实体实验依次验证单实体 GPU命中、单实体 CPU命中、单实体磁盘命中、双实体缓存位置路由、实体失联恢复和网络受限场景。各策略复用同一请求轨迹、模型、缓存初始状态和并发参数。核心指标包含 TTFT、吞吐量、Prefix命中层级、加载字节、路由等待、跨实体流量和尾部延迟。

## 演进顺序

1. 为每个实体建立独立配置、健康状态和 Hardware Profile。
2. 部署每实体 LMCache CPU缓存与磁盘缓存。
3. 建立全局 Prefix目录和 Store、Evict、Clear事件协议。
4. 接入 CPU、磁盘和网络加载代价。
5. 实现实体级容量控制、故障摘除和安全重试。
6. 建立多实体 Benchmark与策略基线。
7. 评估远端缓存传输和 Prefill/Decode分离。
