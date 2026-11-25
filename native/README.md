# DeepSeek-V4.1-Flash 仓库实现微缩版

[executor_repro.py](executor_repro.py) 调用 cann-recipes-infer 的 `OfflineInference.generate()`，用合成图文和测试权重在一张 Ascend 950PR 上执行预填充和解码。文本主干为 6 层、隐藏维度 5120，视觉编码器为 1 层。`--dspark` 增加 3 层草稿模型和主模型验证。

从仓库根目录执行 `bash run_native_repro.sh`，默认依次运行普通生成和 DSpark。也可指定 `normal` 或 `dspark` 单独运行一种模式。[启动脚本](../run_native_repro.sh)

入口按自身路径从 `3rdparty/cann-recipes-infer/` 导入模型和执行器。首次使用需保留主仓的 Git 克隆，以便取得固定版本的子模块。首次运行前需激活 Python 与 CANN 环境。[环境复建说明](native_environment.md)

## 运行

1. 在同一 Bash 终端加载环境复建说明最后一节的 Python 环境和算子包。
2. 从主仓根目录或 `native/` 执行以下命令。
   `--source` 指向仅需包含 `config.json` 和 `chat_template.jinja` 的模型文件目录。首次下载后可复用，已有这两个文件时可直接将 `MODEL_SOURCE` 设为所在目录。

```bash
export NATIVE_DIR="$(git rev-parse --show-toplevel)/native"
export MODEL_SOURCE="${MODEL_SOURCE:-$(git rev-parse --show-toplevel)/build/model}"
revision=314113f8a7dadb388215396ba5fbee2c610a04d6
base="https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/$revision"
mkdir -p "$MODEL_SOURCE"
curl --fail --fail-early --location --retry 2 --remote-name-all \
  --output-dir "$MODEL_SOURCE" "$base/"{config.json,chat_template.jinja}
```

3. 选择一张可用物理卡。入口也会读取 `MODEL_SOURCE` 环境变量，未设置时默认使用主仓 `build/model`。下例使用卡 0，脚本内部访问可见设备 `npu:0`。

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
python "$NATIVE_DIR/executor_repro.py" --source "$MODEL_SOURCE"
python "$NATIVE_DIR/executor_repro.py" --source "$MODEL_SOURCE" --dspark
```

脚本从子模块读取 `models/deepseek_v4_1/config/deepseek_v4_1_flash_rank_8_8ep_multimodal.yaml`，也可以用 `--recipe` 传入绝对路径。运行时覆盖为单 rank、假权重、普通专家路由和单卡 Engram。临时分词器、测试权重和输入均由脚本生成。单卡流程无需下载正式权重、词表、分片索引或图文数据集。固定的框架版本在导入执行器时就会导入 `datasets`，因此 Python 依赖仍包含这个包。当前选择 Ascend C 和普通专家路由，可省去 TileKernels、DeepGEMM 和离线权重转换工具。框架探测可选 TileLang 实现时会打印 `No module named 'tile_kernels'`，本配方继续使用 Ascend C 路径。

模型检查通过且 HCCL 通信组正常关闭后，脚本输出 `"result": "pass"` 并以状态码 0 退出。`calls.vision` 和 `calls.prefill` 均为 1。开启 DSpark 时，`speculative_forwards` 会记录验证轮数。该用例检查执行路径、输出形状与有限非零 logits。真实权重的精度、吞吐与八卡融合通信需另行测试。

在共享 950PR 主机上曾遇到 HCCL 退组时报 `507018`。若设备日志同时显示 `libscatter_aicpu_kernel.so` 缺少 `HcclLaunchAicpuCacheEvictKernel`，请核对该设备实际加载的 AICPU 算子包与主机 CANN 版本。遇到退组异常时，入口会以非零状态退出。

## 在单卡上编译 decode

图模式在 Ascend 950PR、CANN 9.3.0、PyTorch 2.12.0+cpu 和 torch-npu 2.12.0.post2 上验证。从仓库根目录运行下列命令，卡号换成已分配的物理卡。默认配方使用 eager。传入 `--exe-mode npugraph_ex` 后，预填充继续用 eager，执行器在预热期间用 `torch.compile(..., backend="npugraph_ex", fullgraph=True)` 编译 decode。单卡 MoE 在设备上完成本地专家路由。DSpark 会额外编译草稿模型的 decode。

```bash
ASCEND_RT_VISIBLE_DEVICES=0 bash run_native_repro.sh normal --exe-mode npugraph_ex
ASCEND_RT_VISIBLE_DEVICES=0 bash run_native_repro.sh dspark --exe-mode npugraph_ex
```

日志分别位于 `build/logs/normal_npugraph_ex.log` 和 `build/logs/dspark_npugraph_ex.log`。最后一行 JSON 中的 `calls.main_graph_replays` 记录真实请求的主模型设备图回放次数，DSpark 还检查 `calls.draft_graph_replays`。检查 logits 的代码在回放返回后执行。直接调试 Python 时，加载前述环境，在 `executor_repro.py` 的 `run_compiled` 中设置断点即可观察每次调用的输入和输出。

## 用 TileLang 分区执行 decode

`tilelang_ascendc` 和 `tilelang_pto` 需要安装支持 `GraphConfig(partition_policy="ordered")` 的 `torch_tilelang` 及对应目标的 TileLang 工具链。它们使用 Dynamo 捕获完整的主模型或草稿 decode，将支持的浮点运算交给 TileLang，在原位置执行 KV 写入、量化和稀疏注意力等原生操作。视觉编码和预填充使用 eager。

`torch-tilelang` 提交 `bd75b49` 提供本样例使用的有序分区接口。若框架源码与本仓相邻，可在已配置好 Ascend 版 TileLang 和 CANN 的 Python 环境中，从本仓根目录运行 `python -m pip install --no-deps -e ../torch-tilelang`。

本样例在模型注册与注意力元数据中统一设置压缩器的环形缓存长度。压缩比为 2，最长预填充块为 256 个位置，DSpark 每次验证 3 个位置，因此分配 `2 + max(256, 3) - 1 = 257` 个槽。原配方为验证分配 3 个槽，会让历史位置 156 与本轮位置 159 写到同一个槽，两次独立 eager 运行也可能生成不同的结果。缓存修正同时用于 eager、原生图和两个 TileLang 后端。[Compressor V2 的输入约束](../3rdparty/ops-transformer/experimental/attention/compressor_v2/README.md#约束说明)

```bash
bash run_native_repro.sh all --exe-mode tilelang_ascendc
bash run_native_repro.sh all --exe-mode tilelang_pto
```

返回 JSON 的 `tilelang.main` 和 `tilelang.draft` 分别记录主模型与草稿模型的运行情况。`compiled_calls` 从预热完成后开始累计实际请求的分区调用。`kernels` 统计已编译内核，`native_nodes` 统计在分区外执行的 FX 节点。

在单卡 950PR、CANN 9.3 的本样例上，AscendC 和 PTO 均在普通请求中执行 4 个算术分区，共调用 12 次。DSpark 验证主模型执行 4 个分区，草稿模型执行 3 个分区。两个后端在这三张图中均未发生分区回退。

逐步核对输出时，在同一物理卡、相同环境上保存一次匹配的 eager 参考。该参考采用相同的单卡 MoE 路由和 decode 入口。两个对照分支从零初始化 KV 页，在真实请求的每次主模型和草稿模型调用结束后，检查 logits、持久缓存以及最终有效 token。保存的 `normal.pt` 与 `dspark.pt` 可由两个后端共用。

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
bash run_native_repro.sh all --exe-mode eager --write-reference build/reference
bash run_native_repro.sh all --exe-mode tilelang_ascendc --compare-reference build/reference
bash run_native_repro.sh all --exe-mode tilelang_pto --compare-reference build/reference
```

加载[原生执行环境](native_environment.md)后，可从主仓根目录运行配对基准。同一进程固定权重、输入和 MoE 路由，每对请求轮换先后顺序，每种实现预热三次。计时区间覆盖一次完整生成及设备同步。结果单独记录每步 decode、输出 token/s、初始化耗时以及端到端耗时差的配对 bootstrap 区间。AscendC 和 PTO 的运行报告还包含真实请求的分区审计。下例使用已分配的物理卡。

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
mkdir -p build/benchmarks
python native/benchmark_decode.py ascendc --output build/benchmarks/ordered-ascendc.json
python native/benchmark_decode.py pto --output build/benchmarks/ordered-pto.json
python native/benchmark_decode.py npugraph_ex --output build/benchmarks/npugraph-ex.json
python native/benchmark_decode.py profile --output build/benchmarks/normal-hotspots.json
python native/benchmark_decode.py profile --dspark --output build/benchmarks/dspark-hotspots.json
```

`profile` 单独测 eager 中第 1 层 Engram 的浮点 gate 和词表输出投影，记录主机提交时间与 NPU stream 上的事件区间。两者可重叠，耗时不可相加。

### 融合计时的已知限制

本样例使用 `fusion="off"`，通过设备同步和 NPU 事件独立计时。在 950PR、x86 驱动 25.7.rc1 和 CANN 9.3.0 上，如果改用框架的 `fusion="profile"`，默认计时器会在刷新 L2 缓存时收到 `207000`，AscendC 和 PTO 均无法取得本次融合时延。另选 `msprof` 需要 TileLang 提供 `tilelang.profiler.errors.ProfileUnavailableError`。缺少该接口时，框架会记录计时不可用并保留未融合版本。[TileLang #4432](https://github.com/tile-ai/tilelang/issues/4432)
