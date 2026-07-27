# KaReserve

KaReserve 是部署在多个 vLLM 实例前方的 Prefix-aware Router。Router 在短时间窗口内收集请求，根据公共 Token Prefix、各实例的 KVCache 状态和运行负载完成实例分配，然后将请求转发给 vLLM 执行 Continuous Batching。

## 目录结构

```text
serve1/
├── kareserve/                 Router 源码
├── scripts/
│   ├── env/                  环境安装与软件源检测
│   └── debug/                双实例和 Router 联调脚本
├── examples/                 请求样例与 OPT Chat Template
├── runtime/                  服务器运行产物，不进入 Git
│   ├── config/
│   ├── logs/
│   ├── pids/
│   └── reports/
├── .venv-vllm-0.26/          服务器独立运行环境，不进入 Git
├── config.example.json
└── pyproject.toml
```

## vLLM 0.26 环境

服务器环境路径为：

```text
/home/zn/xyz/serve1/.venv-vllm-0.26
```

绝对路径调用能够明确选择版本：

```bash
/home/zn/xyz/serve1/.venv-vllm-0.26/bin/vllm --version
/home/zn/xyz/serve1/.venv-vllm-0.26/bin/python --version
```

交互式操作可以激活环境：

```bash
cd /home/zn/xyz/serve1
source .venv-vllm-0.26/bin/activate
vllm --version
python --version
deactivate
```

激活只会修改当前 Shell 的命令搜索路径。新终端需要重新激活。现有 `/home/zn/vllm_advanced_env` 继续提供 vLLM 0.18.0。

环境安装脚本支持重复执行：

```bash
bash scripts/env/install_vllm_026.sh
```

## 配置

```bash
mkdir -p runtime/config runtime/logs runtime/pids runtime/reports
cp config.example.json runtime/config/kareserve.json
```

`group_block_size` 需要与 vLLM Prefix Cache 的 Block 粒度保持一致。每个 vLLM 实例需要提供 OpenAI API、`/tokenize`、`/metrics` 和独立的 KV Event Endpoint。

## 调试启动

使用 GPU 1 启动单个调试实例：

```bash
GPU_IDS=1 ./scripts/debug/start_vllm_cluster.sh
```

使用 GPU 0 和 GPU 1 启动双实例：

```bash
GPU_IDS="0 1" ./scripts/debug/start_vllm_cluster.sh
```

Router 使用端口 8090：

```bash
KARESERVE_CONFIG_PATH=/home/zn/xyz/serve1/examples/config.single-node.json \
  ./scripts/debug/start_router.sh
```

停止本项目启动的调试进程：

```bash
bash scripts/debug/stop_debug_cluster.sh
```

运行日志位于 `runtime/logs/`，PID 文件位于 `runtime/pids/`。请求样例位于 `examples/smoke_request.json`。

Router 路由状态地址为：

```text
http://127.0.0.1:8090/routing/state
```

## 调度边界

KaReserve 负责请求窗口聚合、公共 Prefix 分组、KVCache 与负载感知的实例选择和 HTTP 转发。vLLM 负责 Continuous Batching、Paged KVCache 和模型执行。LMCache 通过兼容的 vLLM KVConnector 接入外部 KVCache 存储与加载。
