# 系统架构

## 组件边界

KaReserve位于客户端与完整vLLM实例之间。Router管理集群级选点，vLLM管理实例内部调度，LMCache管理GPU之外的KVCache。

请求进入Router后，Tokenizer生成与vLLM一致的Token ID；Request Pool按时间上限或数量上限形成一个路由窗口；Tracker与LMCache Lookup生成请求到各节点的Prefix命中状态；Policy计算窗口内分配；Router按目标实例并发转发独立HTTP请求。vLLM随后执行Continuous Batching。

窗口内的Prefix分组属于路由约束。共享Prefix的请求倾向于进入同一实例。该分组不等同于vLLM执行Batch，也不改变请求协议。

## 状态来源

GPU缓存目录来自vLLM KV Event。Tracker保存每个实例的Block链、Token序列和事件序号，Replay端点补齐Router断连期间仍被发布端保留的事件。vLLM进程代次变化会清空对应实例的GPU目录。

LMCache状态来自项目扩展的只读查询接口。查询接口使用LMCache的Token Hasher生成ObjectKey，直接检查主机内存对象状态，并通过L2 Adapter检查文件系统或对象存储。Router对一个窗口中的请求执行批量查询，相同`cache_domain_id`的vLLM实例共享查询结果。该结果表示查询时刻的缓存状态，目标vLLM Connector仍然执行最终Lookup和锁管理。

实例负载来自vLLM`/metrics`和Router本地在途记录。路由输入包含vLLM运行队列、等待队列、GPU KVCache使用率、Router已分配工作量和请求所需新增Block数。缺失指标保持未知状态，Policy不会把未知值解释为零负载或满负载。

## 路由代价

每个节点的执行路径由GPU命中和LMCache连续命中共同决定。LMCache先锁定主机内存中的连续前缀，再从L2补齐后续Chunk。Policy分别计算L2到主机内存、主机内存到GPU和剩余Prefill成本，然后比较不同节点的总成本。没有外部命中的节点执行完整剩余Prefill。

```text
request cost
= prompt path cost
+ expected decode work
+ vLLM queue and Router inflight work
+ GPU KVCache capacity pressure
```

硬件 Profile 使用 KVCache 字节数、介质带宽和固定延迟计算传输时间。`host_memory_to_gpu_bandwidth_gbps`描述主机运行内存到GPU显存的有效带宽；`prefill_ms_per_token`描述当前模型的实测Prefill时间。Profile缺少完整实测值时，Policy使用归一化工作单位。磁盘路径包含磁盘到主机内存与主机内存到GPU两段成本。

`windowed_prefix`先按公共Token Prefix形成逻辑组，再选择组内总代价最低的共同节点。`window_ms=0`关闭聚合等待，同时保留相同的状态采集与策略实现。

## 一致性边界

GPU目录依赖KV Event连续性。序号缺口无法通过Replay恢复时，节点的GPU目录状态标记为`degraded`。LMCache查询失败时，对应缓存域标记为`degraded`，当前窗口按外部缓存Miss处理。Router状态接口分别输出GPU目录状态和LMCache目录状态。

缓存查询与请求抵达vLLM之间存在时间差。LMCache可能在这段时间发生淘汰，vLLM Connector负责最终正确性。Router查询只用于成本估算，不持有跨组件缓存租约。

## 多主机部署

每台物理主机运行本地vLLM和LMCache服务，并拥有独立GPU、主机内存和文件系统。Router配置使用一个`cache_domain_id`关联共享同一LMCache的vLLM实例，使用`cache_domains.<id>.http_url`定位各主机的查询接口。Policy优先利用目标实体已经持有的缓存。当前数据面不提供跨主机GPU迁移，远端对象存储通过LMCache L2 Adapter进入成本模型。
