# 系统架构

## 执行边界

KaReserve 管理集群级请求选点，vLLM 管理实例内调度，LMCache 管理 GPU 之外的 KVCache。Router 对每个请求独立执行 Tokenize、缓存状态查询、候选构造、策略计算和 HTTP 转发。Router 不聚合请求，也不声明逻辑组对应 vLLM 的实际执行批次。

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

排队成本使用两个状态来源。vLLM 的等待请求数提供已进入引擎的队列状态；Router 的在途工作量覆盖指标刷新前已经转发的请求。在线估计器使用实际首输出路径减去预计 Prompt 路径得到排队样本，并按节点更新“单位预留工作对应的排队毫秒数”。该估计器不绑定模型名称，模型和硬件变化后会重新积累样本。

GPU Block 容量优先作为候选准入条件。所有候选节点的容量快照均不足时，策略保留最低成本节点，以处理指标滞后。高使用率容量成本用于区分仍可容纳请求的候选节点。

## 策略基线

`round_robin`按固定顺序分配请求；`gpu_prefix_load`只使用 GPU Prefix 路径并复用同一排队和容量逻辑；`tiered_completion_time`增加外部缓存介质路径。该分层关系使实验能够分别测量 GPU Prefix 感知和分层介质感知的收益。

## 反馈与一致性

流式响应的首个有效输出用于更新在线排队模型。非流式响应缺少准确首输出时刻，因此只记录完整响应时间。Router 重启会清空在线排队样本，稳定实验需要预热阶段。

LMCache 查询与请求抵达 vLLM 之间存在时间差。vLLM Connector承担最终缓存正确性，Router 查询只参与成本估计。文件系统目录能够确认 L2 对象存在，当前接口无法确认 Linux 页缓存驻留状态，因此文件系统成本使用冷读性能档案。

## 多实体部署

每台物理主机拥有独立 GPU、主机运行内存和文件系统，并运行本地 vLLM 与 LMCache 服务。Router 使用`cache_domain_id`关联缓存域，使用各缓存域的查询地址获取分层状态。跨主机 GPU KVCache 迁移当前不在数据路径中；远程对象存储通过 LMCache L2 Adapter进入成本模型。

## 后续研究边界

多请求联合分配需要独立的聚合策略和执行收益证据。现有 vLLM 接口不会保证窗口内首次 Prefix Miss 只计算一次。该方向在受控实验确认收益条件后再进入主路由路径。
