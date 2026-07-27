# KaReserve配置边界

## 配置层级

KaReserve使用Router JSON、实验Manifest和进程环境变量三个层级。Router JSON描述服务拓扑和路由算法；实验Manifest描述一轮实验；Shell脚本把Manifest解析结果映射为进程环境变量。正式实验入口为`scripts/experiment/run_experiment.py`，手工设置环境变量只用于调试。

```text
实验Manifest
  ├─ runtime         GPU、模型、LMCache和显存比例
  ├─ benchmark       数据集、请求数、到达率、并发和seed
  ├─ router_config   Router JSON路径
  └─ router_overrides策略与窗口消融
          ↓
运行脚本生成环境变量
          ↓
Router JSON提供节点与策略完整配置
```

## Router JSON

`nodes`定义每个vLLM实例。`port`是OpenAI HTTP API端口；`kv_events_endpoint`是该实例发布GPU KVCache元数据事件的ZMQ端口。GPU0调试实例使用HTTP 8101和KV Event 5557，两个端口属于同一个vLLM进程。

`tokenizer_node_id`显式指定Router调用`/tokenize`的实例。所有实例加载同一个模型和Tokenizer时会生成相同Token ID。固定节点保证Token化来源明确。该节点的CPU Tokenizer吞吐需要单独监控。

`routing`定义窗口、窗口容量、策略权重和KVCache容量阈值。`hardware_profile`记录硬件实测结果。当前`windowed_prefix`策略没有使用H2D带宽参数。

## 实验Manifest

`configs/experiments/`中的JSON是一轮实验的权威输入：

```json
{
  "name": "prefix-routing",
  "router_config": "configs/router.two-node.json",
  "repeats": 3,
  "runtime": {
    "gpu_ids": [0, 1],
    "enable_lmcache": true,
    "lmcache_l1_size_gb": 32,
    "model": "/home/zn/llm_models/opt-1.3b",
    "model_name": "kareserve-opt-1.3b",
    "dtype": "half",
    "gpu_memory_utilization": 0.5
  },
  "router_overrides": {
    "policy": "windowed_prefix",
    "window_ms": 2
  },
  "benchmark": {
    "dataset_name": "timed_trace",
    "dataset_path": "runtime/datasets/prefix-trace.jsonl",
    "num_prompts": 256,
    "request_rate": 100,
    "max_concurrency": 64,
    "seed": 0
  }
}
```

运行器为每轮实验保存`run-manifest.json`，该文件包含Manifest、解析后的环境变量、Git提交和关键包版本。

## 环境变量映射

以下变量构成手工调试接口：

| 变量 | 作用 | 默认值 |
|---|---|---|
| `GPU_IDS` | 启动的GPU编号 | `1` |
| `KARESERVE_CONFIG_PATH` | Router JSON | `configs/router.single-node.json` |
| `KARESERVE_MODEL` | vLLM模型路径 | `/home/zn/llm_models/opt-1.3b` |
| `KARESERVE_MODEL_NAME` | OpenAI API模型名 | `kareserve-opt-1.3b` |
| `KARESERVE_DTYPE` | 模型数据类型 | `half` |
| `KARESERVE_GPU_MEMORY_UTILIZATION` | vLLM显存使用比例 | `0.5` |
| `KARESERVE_ENABLE_LMCACHE` | LMCache功能开关 | `1` |
| `LMCACHE_L1_SIZE_GB` | LMCache CPU缓存容量 | `32` |
| `KARESERVE_POLICY_OVERRIDE` | 策略消融覆盖 | 空 |
| `KARESERVE_WINDOW_MS_OVERRIDE` | 聚合窗口覆盖 | 空 |
| `KARESERVE_DATASET_NAME` | vLLM Benchmark数据集类型 | `prefix_repetition` |
| `KARESERVE_DATASET_PATH` | 外部数据集路径 | 空 |
| `KARESERVE_NUM_PROMPTS` | 请求数量 | `128` |
| `KARESERVE_REQUEST_RATE` | 目标请求率 | `20` |
| `KARESERVE_MAX_CONCURRENCY` | 最大在途请求数 | `64` |
| `KARESERVE_SEED` | 数据与到达过程随机种子 | `0` |
| `KARESERVE_TEMPERATURE` | 采样温度 | `0` |
| `KARESERVE_IGNORE_EOS` | 固定输出长度 | `1` |

启动脚本会校验GPU编号与Router端口映射。缺失Router JSON、重复节点ID、错误Tokenizer节点和GPU端口不一致都会使启动失败。

## 配置优先级

正式入口使用以下优先级：

```text
Manifest显式值
→ 运行器生成的环境变量
→ Shell脚本默认值
→ Router JSON完整路由字段
→ policy与window环境变量覆盖
```

Manifest保存每轮差异，Router JSON保持拓扑与策略结构稳定。
