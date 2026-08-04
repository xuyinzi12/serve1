# 系统架构

## 执行边界

KaReserve 管理集群级请求选点，vLLM 管理实例内调度，LMCache 管理 GPU 之外的 KVCache。Router 完成 Tokenize 后把请求交给 Request Pool。Request Pool 收集同一事件循环调度周期内已经就绪的请求，不设置固定毫秒窗口。Router 对规划组批量查询缓存状态，联合计算分配，原子预留资源，再分别转发 HTTP 请求。规划组不声明 vLLM 的实际执行批次。

GPU 缓存目录来自 vLLM KV Event。Tracker 保存各实例的 Block 链和事件序号，Replay 端点用于恢复 Router 断连期间保留的事件。vLLM 进程代次变化会清空对应实例的 GPU 目录。

LMCache 状态来自项目扩展的只读查询接口。查询直接检查主机运行内存对象状态和 L2 Adapter，返回连续命中的 Prefix 长度与介质。同一`cache_domain_id`的 vLLM 实例共享查询结果。vLLM Connector 在执行阶段完成最终 Lookup、锁定和加载。

## 路由代价

主策略`tiered_completion_time`使用统一成本计算候选实例：

```text
request cost
= prompt path cost
+ queue cost
+ capacity cost
```

Prompt 路径成本由 GPU 命中长度、LMCache 连续命中长度、介质传输和剩余 Prefill 决定。主机内存路径包含主机内存到 GPU；文件系统路径包含文件系统到主机内存和主机内存到 GPU。完整 Miss 执行剩余 Prefix 的 Prefill。

排队成本使用两个状态来源。vLLM 的等待请求数提供已进入引擎的队列状态；Router 的在途工作量覆盖指标刷新前已经转发的请求。在线估计器使用实际首输出路径减去预计 Prompt 路径得到排队样本，并对近期样本执行最小二乘估计。节点样本不足时使用集群样本，节点样本充分后使用节点自身结果。该估计器不绑定模型名称，模型和硬件变化后会重新积累样本。

GPU Block 容量优先作为候选准入条件。所有候选节点的容量快照均不足时，策略保留最低成本节点，以处理指标滞后。高使用率容量成本用于区分仍可容纳请求的候选节点。

## 策略基线

`round_robin`按固定顺序分配请求；`gpu_prefix_load`只使用 GPU Prefix 路径并复用同一排队和容量逻辑；`tiered_completion_time`增加外部缓存介质路径。该分层关系使实验能够分别测量 GPU Prefix 感知和分层介质感知的收益。

## 联合分配

Route Planner 为规划组构造请求到实例的成本矩阵。策略先处理可选节点较少的请求，再处理最佳节点与次优节点成本差较大的请求。每次分配都会更新规划组内的虚拟在途工作量和剩余 GPU Block。该顺序把稀缺缓存节点留给依赖程度较高的请求，并让候选范围较宽的请求承担负载分流。

Request Pool 的关闭由请求就绪状态驱动。首个请求出队后，工作协程让出一次事件循环，再一次性取出队列中已经就绪的请求。队列为空时立即关闭规划组；队列达到`max_planning_group_size`时保留剩余请求进入下一组。该机制不会为低 QPS 单请求设置固定等待时间。

## 反馈与一致性

流式响应的首个有效输出用于更新在线排队模型。非流式响应缺少准确首输出时刻，因此只记录完整响应时间。Router 重启会清空在线排队样本，稳定实验需要预热阶段。

LMCache 查询与请求抵达 vLLM 之间存在时间差。vLLM Connector承担最终缓存正确性，Router 查询只参与成本估计。文件系统目录能够确认 L2 对象存在，当前接口无法确认 Linux 页缓存驻留状态，因此文件系统成本使用冷读性能档案。

## 多实体部署

每台物理主机拥有独立 GPU、主机运行内存和文件系统，并运行本地 vLLM 与 LMCache 服务。Router 使用`cache_domain_id`关联缓存域，使用各缓存域的查询地址获取分层状态。跨主机 GPU KVCache 迁移当前不在数据路径中；远程对象存储通过 LMCache L2 Adapter进入成本模型。

## 后续研究边界

现有 vLLM 接口不会保证规划组内首次 Prefix Miss 只计算一次。联合策略只使用已有缓存位置、负载和容量信息，不计入执行中 Prefix 的潜在共享收益。Prefix 副本控制需要后续访问概率和淘汰状态，当前只进入观测数据。
