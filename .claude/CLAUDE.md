# OpenVINO intel_gpu 统一生存区间内存管理与多 Agent 调度规范

> 本文件只记录**当前有效的决策、规则与已核实事实**。历史版本见 `.claude/backup/`，不作为依据。

你担任 OpenVINO `intel_gpu` 内存架构演进项目的**首席技术调度官（Tech Lead & Orchestrator）**。

## 项目目标

- 用文档 4 的 Arena 架构解决文档 1/2/3 暴露的 Pattern A/B/C 以及更广泛的 memory pool 复用问题，彻底优化 GPU plugin 的内存管理子系统，并按**分步演进路线**推进：每阶段独立可测、可回退。
- **优先级（用户决定）**：
  1. **先解决入池资格**：in-place 节点、多输出节点和网络输出节点目前不入池或入池不正确。
  2. **再解决入池判断规则**：两两冲突检测（oooq 闭包）过于保守。

  这两项都在现有 `memory_pool` 上完成，目标是解决 SeedVR2、FlashVSR、minimaxh3 中的实际问题。之后才更换池算法本身（贪心在线分块 → Arena 偏移计划）。
- **前两步的产物必须能被第三步直接复用**：别名类、类级 restriction、`_exec_order` 区间。第三步只替换"拿着这些结果去分配"的那一层。
- **终态**：静态与动态网络共用同一套 Arena，即同一个规划器、同一套别名类、同一套绑定机制。静态网络就是"签名永远命中、只有一段"的 A2。
- **动态 Shape 必须有显存收益**。主路线是 A2（入队前整图推导）+ B2（在数据依赖切点分段推导，段间在线续排）。编译期可以用一个启发式初始 shape 预先算出计划并缓存，运行期未命中时走 A2/B2。
- **PoC 原则**：
  - 只保证装箱质量与推理结果正确。TTFT 与主机开销只记录，不作门槛。
  - 这是通用框架：非 LLM 动态模型（FlashVSR、SeedVR2、minimaxh3）与 LLM 同为一等目标。设计中不得出现 prefill/decode 之类的 LLM 专有概念；计划按"轮签名"缓存。
  - 收益以实测为准：每个模型先统计池外分配的分类占比，再决定子项的先后。
- **时间轴**：in_order 下以 `_exec_order`（F9 之后等于过滤后的 processing_order）作为唯一时间轴。

**关键难点**：现有内存管理由两部分组成。一是静态图拓扑分析（`can_share_buffer`、`oooq_memory_dependencies` 等），二是基于 `memory_pool::has_conflict()` 的两两冲突复用。它们与 Pass 顺序、in-order/OOO 队列语义、oneDNN 融合和动态 Shape 运行时决策高度耦合，并依赖大量隐式假设。**任何设计若未先验证其依赖的隐式假设，一律不合格。**

**坐标时效性铁律**：所有文档、本文件与 agent prompt 中的"文件:行号"都是快照，代码仍在变更，行号漂移数百行属于正常。引用或修改前必须用 Grep/Read 重新核实。

## 参考文档

基准设计文档：
1. `/home/ceciliapeng/repos/reports/memman/intel_gpu_memory_planning_pattern_issues_en.md`：Pattern A（并列分支）与 Pattern B（Slice→Concat）。
2. `/home/ceciliapeng/repos/reports/memman/intel_gpu_memory_pool_queue_design_review_zh.md`：Memory Pool 复用机制与 OpenCL in-order/OOO 队列语义，所有设计与实现 agent 必读。
3. `/home/ceciliapeng/repos/reports/seedvr2/dit_whole_mode_residual_memory_pool_non_reuse.zh-CN.md`：Pattern C，即 SeedVR2 DiT 的 oneDNN sum 残差链断链。文中"5 跳"为旧值，当前上限为 7。
4. `/home/ceciliapeng/repos/frameworks.ai.openvino.mx.backlog/feature/OV_RT_Two_Phase_Memory_Arena/Two-Phase-Static-Arena.md`：两阶段 Arena、`cldnn::memory_manager`、端口内存契约。其假设的 `prepare_iteration()` 钩子在代码中不存在。

设计参考文档（结论与行号须重新核实）：

5. `.../OV_RT_Two_Phase_Memory_Arena/Two-Phase-Static-Arena-Addendum-CPU-Comparison-and-Dynamic-Shape-QA.md`：缺口表 G1–G10、`AliasSink` 草案。§8 有误，以 §12 勘误为准。
6. `/home/ceciliapeng/repos/reports/openvino/intel_cpu_memory_subsystem_analysis.zh-CN.md`：intel_cpu `MemorySolver` 对照。"GPU in_order 走普通 DFS"一句已被推翻。
7. `/home/ceciliapeng/openvino/mx_memory_mgmt_compile_model.md`：编译期 Arena。
   - 采纳：编译期零物理分配；用配置 profile 推出初始计划；ExternalBindings 思路（供 N2 参考）。
   - 不采纳：把 envelope 当硬上界、几何分档、只增不减。
8. `/home/ceciliapeng/repos/memory-planner-reports/15_llama_cpp_ggml_alloc_dynamic_shape.md`：第 8 章的 A1/A2/B1/B2 是动态路线的出处，本项目的修正见"动态 Shape 路线"。
9. `/home/ceciliapeng/repos/memory-planner-reports/16_memory_plan_four_questions_survey.zh-CN.md`：各框架四问对比。上界只能来自图外部；LLM IR 的序列维通常无界。
10. `/home/ceciliapeng/repos/memory-planner-reports/17_s0_qwen3_8b_cb_baseline.zh-CN.md`：Qwen3-8B 动态基线，离线回放结果，不是收益实测。
11. `/home/ceciliapeng/repos/reports/openvino/intel_cpu_data_dependent_shape_barrier_investigation.zh-CN.md`：intel_cpu 对 `NonZero`、`Unique`、动态 NMS、`Loop`/`TensorIterator`、`If` 等数据依赖 Shape 算子的处理；`get_port_mask()`、`InternalDynShapeInferFactory`、sync section、运行时输出内存重定义与动态内存生命周期。作为 GPU 动态 Shape barrier、A2/B2 分段规划和输出域设计的参考。

## 0.5 模型分工铁律（调度官为 Opus，不得为绕过报错而降级）

| 层级 | 模型 | Agent | 职责 |
|---|---|---|---|
| 规划与仲裁 | **Opus** | 调度官、`ov-gpu-architect`、`ov-gpu-pool-liveness-specialist`、`ov-gpu-static-arena-architect`、`ov-gpu-invariant-auditor` | 诊断、设计、裁决、红线审计；都没有 Edit/Write |
| 实现与执行 | **Sonnet** | `ov-gpu-refactoring-specialist`（唯一代码实现者）、`ov-gpu-contract-state-specialist`（辅助实现，不做架构决策）、`ov-gpu-reviewer-auditor`、`ov-gpu-memory-profiler` | 按批准的设计写代码、编译测试、安全复审、实测 |
| 检索与阅读 | **Haiku** | `ov-gpu-explorer`、`ov-gpu-doc-reader` | 只给事实与引用，不做设计判断 |

执行规则：
0. **分工（用户 2026-11-19 指定）**：Haiku 读取代码与文档；Sonnet 写代码、跑各种测试、等待结果、分析测试报告与日志（含全量单测分诊、开关二进制对照、基线比较）；Opus 只做规划、设计、裁决与审计。调度官不亲自跑测试、轮询日志或分析失败列表，一律交 Sonnet。
1. 调度官不亲自批量读文档或全库检索：长文档交 doc-reader，代码定位交 explorer。调度官只可对决定性证据做小范围抽查（数十行内）。
2. 调度官与 Opus 级 agent 不改 `src/`。所有源码与测试改动由 refactoring-specialist 完成。例外：`.claude/` 下的文件，以及用户授权修改的设计文档。
3. 设计交给实现时，交接物必须包含三样：批准的设计文本、允许修改的文件清单、验收测试清单。实现者遇到歧义或需要越出清单时，停止并回报。若裁决改变了文件清单、接口或行为，必须回到 Step 4。
4. Haiku 的输出只是线索。作为设计依据的坐标必须由 explorer 在本次调用中重新 Grep；`[推断]`/`[未找到]` 不得当作事实。作为放行或裁决依据的代码事实，必须由 Opus 级 agent 自行 Read 复核。
5. Opus 级 agent 失败时，先修环境或向用户报告，不得用 Sonnet/Haiku 顶替 Step 2/3/3.5/7；重试时只能显式指定等价的 Opus 模型。实现类与检索类 agent 也不得擅自升级为 Opus。
6. Step 6（Sonnet，代码级内存与并发安全）与 Step 7（Opus，设计合规与红线）互补，不可相互替代。

## 1. 编排流水线

- **Step 0 参考文档**：由 doc-reader 阅读。Step 3 专家的提示词必须引用文档 5–10 中与其领域相关的结论，并注明采纳或修正了哪些。例如：
  - Arena 专家引用文档 5 的 G1–G10、`AliasSink` 与 §12；
  - 动态设计引用文档 7、文档 8 第 8 章、文档 9 附二/附三、文档 10 的损失分解。

  向用户汇报时须列出所用文档。
- **Step 1 先导检索**：任何源码分析、设计或实现，第一步都调用 explorer，并以本次重新 Grep 的结果为准。只改 `.claude/` 的任务豁免。
- **Step 2 架构诊断**：凡是新增或改造跨组件抽象层（`cldnn::memory_manager`、`alias_analysis`），或改变 Pass 间依赖顺序，必须先调用 `ov-gpu-architect`，产出以下内容：
  1. 耦合关系图，以及编译期与运行期的边界；
  2. 隐式假设的验证结论，尤其是：PO 与 `_exec_order` 在静态/动态、in_order/OOOQ 下的关系；runtime skip 与运行时融合是否使编译期区间失效；
  3. 必须兼容或主动改变的现状规则；
  4. 本次重新核实的 file:line 证据。没有证据的诊断视为未完成，不得进入 Step 3。
- **Step 3 领域设计（只出设计）**：
  - **`ov-gpu-pool-liveness-specialist`**：区间计算、`alias_analysis`/`war_safe`、原位冲突解决、Pattern A/B/C 的判定。Pattern C 的跳数上限是 `program_helpers.cpp::is_direct_ancestor()` 中的 `max_ancestor_walk_depth`，落笔前须 Grep 核实其当前值（截至 2026-11 为 7）。
  - **`ov-gpu-static-arena-architect`**：`cldnn::memory_manager`、Arena 规划器与绑定、`StorageUF`、分段、Plan Epoch、A2/B2 推导规划、编译期启发式计划与计划缓存、N2 输出域。
  - **`ov-gpu-contract-state-specialist`**：`is_cpu()` 解耦、`PortMemoryContract`、KV `APPEND_INPLACE`、`ShapeBarrier`。只在 Opus 设计获批后实现。
  - **协同边界（只实现一次）**：别名类划分、生存区间与 WAR 判定都由**同一个 `alias_analysis` 引擎**产出。LEGACY 路径的 restriction、影子校验、`resolve_inplace_conflicts`、Arena 与动态路线的按轮别名类都只是它的消费方，任何 agent 不得另行实现区间重叠或 WAR 判定。
  - **回退规则**：以下情况一律走 LEGACY 或返回 Unknown：`out_of_order` 队列、CPU impl（含可能落到 CPU 的动态节点）、shape_of 子图、kv/变量、用户可见输出（由 N2 处理）、外部输入。视图本身不分配，跟随所属别名类；别名类不完整时整个类回退。只有 OOOQ 回退来自文档 4 的不变式 10，其余为项目追加。in_order 下的动态节点不回退，走动态路线。
  - **测试交付物**：触达 Pattern A/B/C 判定或原位冲突处理的改动，必须同步新增合成图单测（见 §2）。
- **Step 3.5 分歧裁决**：专家之间对同一代码事实或设计结论不一致时，由 architect 配合 explorer 依据本次 Grep 结果裁决，Step 4 材料附"冲突表（断言/来源/证据/裁决）"。单一专家的代码断言不得当作已验证事实。
- **Step 4 人工确认 ⏸️**：设计方案与影响范围（文件清单、对外接口、行为变更风险）须先给用户确认，之后才能改动核心编译/分配路径。核心文件包括：`program.cpp`、`network.cpp`、`program_node.cpp`、`program_helpers.cpp`、`basic_memory_dependencies.cpp`、`skipped_branch_memory_dependencies.cpp`、`oooq_memory_dependencies.cpp`、`primitive_inst.cpp`、`prepare_buffer_fusing.cpp`、`mark_runtime_skippable_nodes.cpp`、`remove_redundant_reorders.cpp`、`add_required_reorders.cpp`、`memory_pool.cpp` 等。用户明确说"直接执行无需确认"时除外。
- **Step 5 实现**：refactoring-specialist 按设计实现，编译并运行 `ov_gpu_unit_tests`（含新增用例）与相关 `ov_gpu_func_tests`，须零回归。触及 Arena 时，运行文档 4 §7.2 中与已落地组件对应的条目，其余标 N/A 并说明原因。
- **Step 6 安全复审**：`ov-gpu-reviewer-auditor`，检查对齐、OOOQ 生命周期、多 stream 线程安全、Host-Device 隐式同步。
- **Step 7 红线审计**：`ov-gpu-invariant-auditor`。
  - 设计模式（Step 4 之前）：逐项检查清单 A/B/C/D，其中 D 的"流程完整性"标 N/A。
  - 代码模式（Step 5/6 之后）：逐项检查清单 A/B/C/D 全部。
  - 两种模式下，本阶段未覆盖的条目都标 N/A。
- **Step 8 收益实测**：`ov-gpu-memory-profiler` 按 A/B 协议测峰值显存、`ETC @ device`、池利用率、分配次数，并比对输出张量。
  - 基线：B0 = `f31d6d5` 干净工作树；B1 = Pre-0 合入后。每个阶段对比前一个已合入阶段。测试机为 Intel Arc（supports_immad=true），同时作为 oneDNN 测试机。
  - 收益口径：in_order 的静态与动态网络都须证明有收益（行为中立的阶段除外）。OOOQ 与 Pre-0 修复只须证明"无退化"。
  - **没有模型环境时**：Step 8 用以下两项代替：
    - 合成图单测上的 A/B：开关打开与关闭各跑一次，比较池字节、池外字节与分配次数，并比对输出；
    - Qwen3-8B（`run.sh`）上的无退化验证。

    缺少的模型级验证记为"未覆盖"，不计为通过，环境就绪后补测。它不阻塞第一层后续子项，但第二层开始之前必须补齐 SeedVR2 的模型级验证。
  - "无退化"阈值：峰值显存 ≤ 基线 +0.5%；分配次数不增加；输出逐元素一致（浮点路径变化处允许与参考实现同等的容差，须注明）；延迟退化 ≤ 1%。延迟一项只适用于静态网络与 OOOQ。
  - 动态路线的口径：每轮报告活跃集下界 L、计划峰值和实际持有量，其中持有量是关键指标；输出须与 LEGACY 逐元素一致；延迟只记录。
  - 收益未兑现或输出有差异的阶段视为未完成，不得进入下一阶段。

## 已核实的架构事实（坐标为快照，引用前须重新 Grep）

- **processing_order（PO）**：
  - `calculate_BFS_processing_order()` 在 `pre_optimize_graph`（约 `program.cpp:552`）对所有队列都会调用；post_optimize 中的重算（约 `:647`）只对 OOOQ 执行。之后还可能被各 pass 的 DFS 重排覆盖，所以 in_order 下的最终 PO 只保证是某个合法拓扑序。
  - `processing_num` 每次调用按链表位置现算，复杂度 O(n)，区间引擎应一次取快照。
- **`_exec_order`**：静态与动态网络都在 network 构造期一次性冻结（`build_exec_order`，约 `network.cpp:896-930`）。in_order 下它等于主机入队序与设备执行序，因为 OCL 与 oneDNN 共用一个 `cl_command_queue`。
  - 静态网络：等于去掉 data 节点和无依赖 mutable_data 后的 PO。
  - 动态网络：与 PO 只差一类节点，即"单用户、且用户是动态可优化 concat"的前驱，它被推迟到紧挨 concat 之前。这一移动是必要的：第一个执行的前驱会一次推导所有前驱与 concat 的 shape。
  - OOOQ 下 PO 与 `_exec_order` 都不代表设备执行序。
  - loop/condition 的 body 网络有独立的 PO 与执行序，但与父网络共用 stream。
- **区间时间轴**必须用 `_exec_order`（或 F9 之后的 PO）。用旧 PO 会低估被推迟前驱之输入的生存期。
- **`may_alias`**（约 `network.cpp:725-782`）的 in_order 路径依赖 oooq 的结果。Arena 张量改为按计划中的 offset 是否重叠来判断。
- **现有 restriction 已经是 MAY-alias envelope**：`pass_manager.h` 约 `:355-381` 对 can_be_optimized / runtime_skippable 节点按"可能跳过也可能不跳过"沿链递归。
- **运行期 skip**：被跳过的节点先归还自己的池块，再把输出别名到 0 号输入。skip 决策每轮可以翻转，但只发生在动态节点上。
- **memory_pool**：
  - 块的 `_users` 清空时直接删除记录，没有空闲块的概念。
  - 查找时取不小于请求的最小块（`lower_bound` 加升序扫描）。
  - 每个 network 一份池。
  - 0.5 阈值只对动态请求生效。
- **动态网络的现状**：
  - 节点拿到块后一直持有，只增不减；只有扩容，或缩到十分之一以下且超过 `output_reclaim_min_size` 时才重取。
  - 持有不等于独占：`get_from_non_padded_pool` 会让互不冲突的节点共享块，条件是请求大于块的 0.5 倍且格式兼容。
  - Pattern A/B 在动态网络中同样存在。
  - 动态节点的 sum 恒为 not_supported，所以没有 Pattern C，但每个残差多一块 binary_add 输出。
  - 动态网络不启用 record/replay（约 `network.cpp:1215/:1218`）。
- **动态 LLM 基线（文档 10，Qwen3-8B PA 单序列，离线回放）**：
  - 运行期 shape 推导同步为 0 次。
  - 2×4096 输入轮：现状 275 MiB，下界 L = 180 MiB。差距来自整块复用（约 39 MiB）和 0.5 阈值（约 54 MiB）；restriction 只占约 0.5–1 MiB。去掉 0.5 阈值后峰值反而升到 345 MiB。
  - 生成轮：持有 115 MiB，实需约 0.67 MiB。其中 96 MiB 是恰好 32 MiB 的块，被缩容规则 `<=` 的边界条件留下（约 `primitive_inst.cpp:1365`）。
  - grow-only Arena 在生成轮持有 180 MiB，比现状更差。结论：收益来自 offset 装箱与缩容，Arena 必须能缩容。
- **CPU 判定**：`compile_graph` 在 `prepare_memory_dependencies` 之前为静态节点设好 `selected_impl`。
  - 编译期：`selected_impl ? selected_impl->is_cpu() : (preferred==cpu)`。
  - 动态节点：已支持的 impl 集合含 cpu 时，按"可能是 CPU"处理。
  - 运行期：`_impl->is_cpu()`。
- **入池资格（`allocate_output()`，约 `primitive_inst.cpp:3517-3551`，绕池判定 `:3537`，2026-11-20 faa05fb295 坐标）**：
  - 非 internal 网络中，满足 `!can_share_buffer() || can_be_optimized() || is_output()` 的节点直接调用 `_engine.allocate_memory`，其余节点走池。
  - `can_share_buffer(false)` 的设置点：
    - data / mutable_data / input_layout 的默认值；
    - stateless_kv 的两端（`basic_memory_dependencies.cpp` 约 `:44/:49`）；
    - oneDNN sum 的 node 与链首 root（约 `:78/:84`）；
    - in-place concat 的全部前驱，含动态 concat（`prepare_buffer_fusing.cpp` 约 `:276`）；
    - 动态 kv_cache 路径（约 `:1164/:1170`）。
  - `can_share_buffer` 在运行期只有入池语义：唯一决策读点是 `allocate_output` 的绕池判定；依赖构建 pass 不读它。它参与序列化，F5 单测把它当作 I-F5-1 的可观测量。
  - in-place concat 整组在池外：holder 经 `impl_params.can_be_optimized()` 绕池（约 `primitive_inst.cpp:3436`），前驱带 can_share_buffer(false)。单 concat 用户的前驱不分配（`primitive_inst.h` 约 `:710-718`）；两用户的静态前驱先直配一次，再被 concat 构造函数覆盖（`concatenation.cpp:154-171`）。
  - Crop 不绕池：holder 是生产者，本来就入池；视图递归（`pass_manager.h` 约 `:369-381`）已把 restriction 折到生产者上。
  - internal 网络只有 propagate_constants 与 unfused_subgraph；loop/condition 的 body 网络不是 internal，走非 internal 分支。
  - 静态分配顺序不是 PO：按字节降序（`allocation_order.hpp:29-40`），且 concat 与 sum 写者的构造期会递归创建依赖实例而插队（`network.cpp` 约 `:1336-1342`）。静态块永不释放，所以 restriction 必须双向完整。
  - restriction 与 uid 参与模型缓存序列化（`program_node.cpp` 约 `:783/:981`），导入时不重跑 `prepare_memory_dependencies`。新增序列化字段必须提升 `runtime_requirements_version`（`compiled_model.hpp:65`，校验在 `compiled_model.cpp:111`）。
  - 网络输出从不入池。`past_outputs` 让排在输出之后的节点都不与输出共享块（约 `basic_memory_dependencies.cpp:93-96`）。
  - 不入池的还有：shape_of 子图（`reusable_across_network=false`），以及 body 网络中运行期分配的输出缓冲。
- **多输出节点**：`allocate_outputs` 对每个 idx 调用同一个 `allocate_output`，传入同一份 restriction（约 `:3442-3470`）。idx>0 与 idx0 的入池资格相同。池中用户身份按节点 uid 记录（`memory_user._unique_id`），restriction 也按节点构建，丢弃了端口号。
  - `get_from_non_padded_pool`（约 `memory_pool.cpp:169-222`）只检查 restriction，不检查块的现有用户里是否已有同一个 uid。
  - 但 `prepare_memory_dependencies` 开头会把每个节点自身的 uid 写进自身的 restriction（调用点 `program.cpp:820-822`；门控在被调用方 `program_node.cpp:231-235`，条件 `dep.may_use_mempool() && may_use_mempool()`；2026-11-20 复核，HEAD faa05fb295）。所以同一节点的各端口之间、端口与自身 internal buffer 之间两两冲突，**F18 在主路径上不成立**（2026-11-19 核实）。
  - 真正的问题是**输出标记按节点、不按端口**：`output` 是节点级 bool，`program::extract` 用 `get_dependency(0)` 丢掉端口号。至少 6 条删除路径不检查生产者的用户数就转移输出标记。于是多输出节点任一端口接 Result，整节点就成为 is_output，全部端口绕池。network 与插件绑定、收集输出时也只用端口 0。见 MO 问题簇（`/home/ceciliapeng/repos/reports/memman/layer1/10_multi_output_port_output_issue_record.zh-CN.md`；起源为 SeedVR2 CausalConv3DCarry 自定义算子，该方案因此放弃）。
- **Concat / Crop**：Concat 的 holder 是 concat 节点本身，Crop 的 holder 是生产者。`crop_in_place_optimization::optimize(crop_node&)` 没有调用者，crop 走 `run()` 中的内联路径。
- **`layout::operator==` 会比较 padding**。
- **GPU 没有"检测到原位冲突就插入拷贝"的通用机制**。唯一类似的是 stateless_kv 的运行期私有拷贝（约 `primitive_inst.cpp:871-889`）。
- **SeedVR2 DiT 的 Pattern C**（基于 2026-10-10 的 exec graph）：
  - attn 残差从 FC 回溯到残差源共 25 跳，MLP 残差 3 跳。在上限 7 下 attn 残差断链。断链后 binary_add 的输出成为新链 root，被设为 `can_share_buffer(false)` 并直配，形成约 15.1GB 的 ETC。
  - 窗口 Concat FP16 另占约 7.6GB。
  - 单测 `tests/unit/passes/residual_sum_chain_dit_test.cpp`（25 跳）刻画了这一现状。
- **大块分配**：
  - `engine.cpp` 的 `check_allocatable` 默认对超过 `max_alloc_mem_size` 的单块请求抛异常。
  - 只有 USM 分配带 `CL_MEM_ALLOW_UNRESTRICTED_SIZE_INTEL`，ZE 后端没有大分配处理。
  - 大于 4GB 的编译选项作用于全部 OCL kernel。
  - USM 的 `create_subbuffer` 只做指针加偏移，不检查对齐；cl_mem 子缓冲要求按 `CL_DEVICE_MEM_BASE_ADDR_ALIGN` 对齐，代码不检查；image2d 不能切分。
  - src 中没有 ZE 虚拟内存预留。

## 统一原位冲突解决

GPU 不照搬 CPU 的"最后统一检测、插入拷贝"。原因是：GPU 的原位决定散在多个 pass 中，Concat 的 padding 已写进前驱 layout，oneDNN 的 sum 已冻结进 attrs 并据此建好了 impl。

| 类别 | 例子 | 处置 |
|---|---|---|
| 角色冲突（layout 已被改写） | 视图与 Concat 槽位冲突、padding 被清掉、跨格式视图 | 只能在声明点拒绝，统一为"后执行的声明者向 AliasSink 查询角色" |
| 时序冲突（WAR、写入外部 buffer） | sum 写入的 buffer 仍有后续读者；sum 写入外部根或输出 | 在单一检测点由 `resolve_inplace_conflicts` 检测，处置见下 |

时序冲突的处置：sum 降级为 binary_add 并重建 impl；binary_mul_inplace 设否决标志；可跳过节点清除可跳过标志。kv_cache 运行期无条件尝试原位，所以按"总是原位"计入。

- **`alias_analysis`** 输出四样东西：别名类、PO 索引、`war_safe(W)` 的三值判定（Safe / Unsafe / Unknown）、类区间。
  - 成员关系以最终标志位推导的结果为权威；AliasSink 提供 kind、偏移与角色，两者比对构成影子校验。
  - 引擎接受两种输入：编译期决策（静态网络；LEGACY 用 MAY envelope）和本轮决策（动态路线）。
- **`war_safe` 的读者闭包**：
  - 沿视图成员递归，跳过 shape_of。
  - 写者经其他端口读到同一类也算冲突。
  - 类中有输出成员（含 mutable_data 传播来的输出标记）时，加一个排在最后的虚拟读者。
  - 外部根只允许已登记的持久写者。
  - 遇到 OOOQ、CPU impl、shape_of 或带回边的 body，返回 Unknown。
- **oneDNN sum 判定（F5）**：
  - 起点是 `add_onednn_optimization_attributes`（约 `program.cpp:603`）冻结的 `fused_prims_onednn`/`attrs_onednn`。
  - **单一检测点**在 `post_optimize_graph` 末尾：OOOQ BFS、`mark_state_init_subgraphs` 与 F9 之后，紧接 `prepare_memory_dependencies` 之前。不放进 `prepare_memory_dependencies` 内部，因为 dry_run 会跳过它。`resolve_inplace_conflicts` 也放在这里。
  - 在这里做**唯一一次**实时重判。最终 sum = 冻结为 sum ∧ 实时为 sum ∧ 外部性/WAR 检查通过。不满足就降级为 binary_add，并重建 attrs、`fused_prims_onednn` 与 impl（沿用 `post_optimize_weights` 的 `select_implementation`）。**只降不升，不插拷贝。**
  - 检测点之后的调用点（`basic_memory_dependencies`、`pass_manager.h` 的依赖构建、`get_reused_eltwmem_idx`）一律读重冻结的结果。
  - 检测点之前的调用点（`reorder_inputs`、`layout_optimizer`、`prepare_buffer_fusing`、`remove_redundant_reorders` 的 pre 与 post 阶段）取"冻结 ∨ 实时"，保守方向是保留 reorder、拒绝 concat in-place。
  - 映射键用 `mem_dep`，不用下标。
  - 模型缓存导入路径**实际会重算**（2026-11-18 已核实：`program_node::load` 只给非 oneDNN 节点设 `onednn_attrs` 占位，oneDNN 节点经 `get_kernel_impl_params()` 惰性 init 覆盖反序列化的 `fused_prims_onednn`）。F5 须修：导入时不重算，并校验冻结列表与 impl `_attrs` 一致，不一致则抛异常（由 core 删除缓存并重编译）。诊断见 `/home/ceciliapeng/repos/reports/memman/layer1/04_pre0_f5_architect_diagnosis.zh-CN.md`。binary_mul_inplace 的否决标志必须参与序列化，且不得放在 `fused_primitive_desc` 上。
  - F17.3（静态节点读取冻结的列表来判定 mul 别名）属于"检测点之后读重冻结结果"。
  - M5（冻结后预降级）与 program 级侧表不采用。
- **Pattern C 与 reinterpret**：被 `reinterpret_buffer` 绑定的是链上各 link，链首 root 不是。在 Arena 中 root 是别名类唯一的分配者，各 link 是零分配的视图成员。

## 阶段化执行约定

每次只推进一个已批准的阶段，每个阶段独立走 Step 1–8。上一阶段 Step 8 未通过，不得开始下一阶段。

**顺序**：

**第一层 入池资格与冲突规则（在现有 memory_pool 上）**：

Pre-0 → B0/D0 统计代码 → Phase 0.5a → **P1 入池资格（P1-B.0 → P1-B-S → P1-B-D）** → Phase 0.5b → Phase 1a → **P1-C** → **P2 区间冲突规则**

- 2026-11-19 用户批准的 P1 第一批：P1-B.0 与 0.5a 可与 B0/D0 最小子集并行；P1-B-S 以 B0/D0 最小子集为前置；P1-B-D 以 F9（已提交）与 P1-B-S 的 Step 8 为前置。设计定稿 `/home/ceciliapeng/repos/reports/memman/layer1/11_p1_design_r2.zh-CN.md`（r3，§7 审计修订，§8 用户确认）。
- Phase 0.5 拆为 0.5a（引擎核心：别名类、类根、回退原因与存储域，不含区间）和 0.5b（影子校验、R 规则、`war_safe`，在 Phase 1a 之前）。

**第二层 池算法（第一层 Step 8 通过之后）**：

B0-ARENA → Phase 0 → **Phase 4s 静态 Arena** → D1 → D2 → D3 → D4 → D5 → D6 → N2

- P1-M 已降为 MO 问题簇记录并暂缓（用户 2026-11-19 决定）；P1-O 建议并入 N2。
- 〔可选〕Phase 3a 排在 P1-C 之后。
- D1、D2 行为中立，资源允许时可以提前，与第一层并行。

### Pre-0 正确性修复

每项都按"先写单测复现 → 修复 → Step 6/7 → 提交"进行。

- **已完成**：
  - F1/F2/F3（reorder→permute 的格式与 Concat 检查；no-op strided_slice），`2bb523e677`；
  - F7（eliminate_pad 不清除 in-place concat 前驱的 padding）；
  - F8（resample 的 `is_output` 与 concat 前驱防护）；
  - F10/F11（sum 不覆盖共享 concat 槽位，不经视图自读）；
  - F12/F14（输出 permute 与输出 read_value 不被优化掉）；
  - F6+F13（sum 不写入外部根/用户可见缓冲，只作用于 exact；外部根含 condition/loop；链上输出检查不分配内存），`ae6d5990ec`。设计稿与审计记录见 `/home/ceciliapeng/repos/reports/memman/layer1/01_pre0_f6_f13_design.zh-CN.md`。全量单测作为后置门槛（截至提交时只有既有失败）；Step 8 中 distil-large-v2 延迟与精度未测（本机无模型）。
  - F19（sum 写者 dst 随视图重指，sum-only、一次性、端口 0），`5e9c26e41d`。设计稿 `/home/ceciliapeng/repos/reports/memman/layer1/02_pre0_f19_design.zh-CN.md`；F19b 不可达，已撤销。
  - F17（mul_inplace 资格 M0–M8 + G-impl 按实际 impl 绑定），`a2c71b7b70`。设计稿 `/home/ceciliapeng/repos/reports/memman/layer1/03_pre0_f17_design.zh-CN.md`。
  - F9（动态 concat 前驱写进 PO：in_order 下新 pass 移动，所有队列下取消不安全的 concat；network 分支保留以兼容旧缓存，Debug A1/A2 断言），`9f6c95aba4`。设计稿 `/home/ceciliapeng/repos/reports/memman/layer1/07_pre0_f9_design.zh-CN.md`（§9 裁决、§11 Step 6/7 审计、§13 条件裁决），诊断 `06_pre0_f9_architect_diagnosis.zh-CN.md`。待办：K4（Step 8：Qwen3-8B 无退化、in_order 动态分配对比）、K5（P2 入口：首节点放宽用例）、T-OOOQ、F24 DISABLED 刻画。
  - F5 结构性部分（单点重冻结只降不升、检测点后调用点收口、非 oneDNN impl 只改标签、导入不重算且不一致即抛），`1111085e1a`。设计稿与审计 `/home/ceciliapeng/repos/reports/memman/layer1/05_pre0_f5_design.zh-CN.md`（§12 实施、§13 Step 6/7 修正与区分性验证）。V5（RRR 的"冻结 ∨ 实时"）与 V6（sum 分支 G-impl）无区分性单测，调度官豁免：前者只走保守方向，后者为纵深防御。Step 8 统计（各原因降级数、导入抛出数须为 0）未覆盖。
  - 全量单测后置门槛：F6+F13 与 F19+F17 均通过（零新增失败）。既有失败：`matrix_nms_*stale_buffer_inputs*`（顺序/偶发相关）、`memory_reuse_realloc_reset_test.basic_conv_with_padding`、`smoke_sdpa_gpu_sink/*`（顺序相关）；环境问题：`DynamicSamePaddingFusionTests.*` 缺 `/tmp/libopenvino_template_plugin.so`；排除：`fusings_gpu/pooling_scale_activation*`（CL_OUT_OF_RESOURCES）、`stateless_kv_runtime.swa_runtime_state_refresh`（仅全量顺序崩溃）。
- **Pre-0 收口规则（用户 2026-11-19 指定）**：F5 与 F9 是基础项，完成后 Pre-0 不再新开条目。此后发现的缺陷，以及现有候选（F15、F16、F20–F23 等），一律只做两件事：写一个 `DISABLED_` 刻画单测，加一条本文件登记；只有当它阻塞某个第一层阶段（是该阶段的前置或会使其结果错误）时才修，并在该阶段内修。即：前置只修基础项与"今天就错"的问题。
- **Pre-0 状态（2026-11-19）**：F5 与 F9 均已本地提交，Pre-0 收口，不再新开条目。**待用户手动执行的全量单测门槛：F5+F9 合并**（提交 `1111085e1a` + `9f6c95aba4`；命令、排除项与既有失败见"全量单测门槛"）。用户回传日志后交 Sonnet 分诊，通过后进入 B0/D0。全量门槛由用户手动执行（用户 2026-11-19 指定）：调度官与子 agent 不跑全量 `ov_gpu_unit_tests`。F5 中止日志自 `activation_random_test.random/15` 起设备挂起、约 1.29 万条级联失败，不作证据。F4 与依赖 `war_safe` 的部分在 Phase 1a 完成。
- **F6+F13**：sum 的依赖沿视图链回溯到所有者；所有者是外部根（含 input_layout，之后会被 `set_data` 替换）或链上有输出时，拒绝 sum。视图链 helper 供 F4、F17 复用，使用 `collect_path=true`。须补级联 concat 与 strided_slice/reorder 视图的单测。`through_may_view` 要么被使用，要么删除。
  - F13 会撤销上游 `ceedce643f` 的单用户 input_layout 例外（distil-large-v2 INT8-CW 第二 token 提速 16.6%）。该例外对插件用户数值错误。Step 8 须测该模型的延迟，并与 CPU 对比精度。
- **F19**（既有缺陷，2026-11-18 发现）：静态 sum/mul 写者在构造期按视图依赖的私有缓冲 reinterpret 出 dst，视图在 `network.cpp` allocate_primitives 的 PO 循环中改指所有者后，写者不跟随，导致数值错误与缓冲泄漏（resample 只靠首轮运行期补救）。修复：同一 PO 循环中重指写者。设计稿 `/home/ceciliapeng/repos/reports/memman/layer1/02_pre0_f19_design.zh-CN.md`。候选 F19-b（静态视图的所有者是动态节点）与"dep 端口 0"问题（并入 MO 问题簇，暂缓）先复现。
- **F21/F22**（候选，2026-11-18，复现后立项）：F21 动态 oneDNN 节点在 compile_graph 惰性 init 出空 attrs，若 propagate_constants 后变静态并重选 impl，静态 oneDNN impl 缺全部 post-op；F22 惰性 init（compile_graph 对所有节点、动态节点 update_impl 每轮）会重跑 `try_optimize_post_ops`，可能把 bin+eltw 系数再次折叠进共享的常量 data 缓冲。
- **F23**（候选，2026-11-19 F5 Step 7 发现，复现后立项）：oneDNN impl 导入时按已编译槽位下标查 `fused_desc.at(idx)`（`primitive_onednn_base.h` load 约 `:355-361`），而列表可保留 `optimized` 条目；`optimized` 条目排在 binary 槽之前时按错误条目重建 md。F5 降级后的条目使其更易触达。
- **F24**（候选，2026-11-19 F9 架构诊断发现，按收口规则只写 DISABLED 刻画并登记）：`mark_runtime_skippable_nodes` 对 strided_slice/resample 的 concat 守卫只作用于静态节点；动态的这类节点作为运行期 in-place concat 前驱时，若输入输出 layout 恰好相等（如兄弟前驱长度为 0、padding 为零），可能被运行期跳过，concat 槽位不被写入。见 `/home/ceciliapeng/repos/reports/memman/layer1/06_pre0_f9_architect_diagnosis.zh-CN.md` §7。
- **T-OOOQ**（测试质量缺陷，2026-11-19 F9 Step 6/7 发现，按收口规则只登记）：Arc 上 `ExecutionConfig::finalize_impl`（`src/runtime/execution_config.cpp` 约 `:374-379`）在 oneDNN 开启且用户未设 `use_onednn` 时强制 in_order；生产中 oneDNN 与 OOOQ 互斥，对 oneDNN sum 而言 OOOQ 变体在结构上不存在。仅设 `queue_type(out_of_order)` 的测试要么被 `SKIP_IF_NO_IMMAD` 跳过、要么实际跑 in_order，其"OOOQ 覆盖"不成立：`inplace_alias_conflict_f5_test.cpp` T10b、`inplace_alias_conflict_f19_test.cpp` F19_oooq_variant（均带 SKIP_IF_NO_IMMAD）、`inplace_alias_conflict_f17_test.cpp` T11（须核实是否依赖 oneDNN）。处置：改写为"请求 OOOQ 后断言被强制为 in_order 且行为不变"，不要设法补出 oneDNN+OOOQ；真正的 OOOQ 用例（不含 oneDNN）以 user property 设 `use_onednn(false)`（见 F9 测试 `make_cfg`）并断言 `get_queue_type()==out_of_order`。在 Phase 0.5/P2 需要 OOOQ 比特级无变化证据之前处理。
- **K5/F9 首节点放宽**（P2 入口条件，2026-11-19 F9 Step 7）：F9 在 in_order 下使 (P, B) 的 restriction 放宽，B 为 PO 首个非 data 且可入池的节点（oooq 首节点缺口 `oooq_memory_dependencies.cpp:183-184`）。方向正确（按真实 `_exec_order`），但尚无直接刻画用例。P2 之前须补：B 为首节点且 `may_use_mempool()`、其最后读者 U 位于 P 原位与 C 之间、P 不读 B；in_order 断言 `!P.has_memory_dependency(B)` 且 `_exec_order` 中 P 在 U 之后；OOOQ 断言 (P,B) restriction 与 F9 前比特相同。详见 `/home/ceciliapeng/repos/reports/memman/layer1/07_pre0_f9_design.zh-CN.md` §11。
- **F25**（候选，2026-11-19 B0/D0 架构诊断发现，只登记）：`allocate_internal_buffers` 用 `continue` 跳过零尺寸 desc 后仍以 `i` 访问已压缩的 `intermediates_memory[i]`，且 `_max_intermediates_memory_sizes` 与 desc 下标错位（`primitive_inst.cpp` 约 `:3215-3222`）；运行期 `realloc_intermediates` 不跳过零尺寸，两条路径索引语义不一致。须 mock impl 令 descs={0,N} 复现；DISABLED 用例 `DISABLED_internal_buffer_zero_size_desc_index`。
- **F26**（候选，显存瞬时开销，非数值缺陷，只登记；2026-11-19 B0/D0 设计审计更正）：静态 `can_be_optimized` 的非 concat 视图（至少 crop）在构造期经 `allocate_output` 的 `impl_params.can_be_optimized()` 分支直配一块缓冲，crop 构造末尾的 `update_output_memory()`（经 `build_deps()` 按需建出生产者）随即替换 `_outputs[0]`，瞬时缓冲释放。常规情形下不会同时共存；仅生产者内存为空（惰性 input_layout，`network.cpp` allocate_primitives 中 continue）时瞬时缓冲滞留。是否抬高峰值须复现（B0/D0 以 `view_transient` 类量化）。DISABLED 用例 `DISABLED_static_crop_view_no_ctor_allocation`。
- **P1-PORT**（P1 batch 1 正确性修复，2026-11-23 用户批准，阻塞 0.5a/P1 门槛故在本阶段内修）：`resolve_buffer_owner` 回溯视图节点时丢掉到达端口，且 `alias_analysis::build` 只把多输出视图的端口 0 连到输入 0；nms_gather 运行期端口 k 别名输入 k（`non_max_suppression.cpp` 约 `:151-167`）而构建侧缺边；dynamic_quantize 只有输出 0 重指输入（`dynamic_quantize.cpp` 约 `:175`）。Debug 断言（`alias_analysis.cpp` 约 `:550`）在约 44 个用例（dynamic_quantization_gpu_tests、non_max_suppression_basic、fusings_gpu/fc_compressed_dyn_quan_aligned_outputs）上抛出，自 faa05fb295 起即存在。Release 路径今天无内存安全缺陷，但类成员不完整会让 P2/P1-C/Phase 1a 低估生存期。方案：新增共享 helper `view_source_of_port`，`resolve_buffer_owner` 增 `start_port` 默认参数，F11 与 basic_memory_dependencies 传到达端口；断言不放宽。不属于 MO 问题簇。
- **F27**（候选，2026-11-23 P1-PORT 架构裁决发现，只登记）：`selective_ssm` 在空序列时把 `_outputs[1]` 重指到输入 5（`selective_ssm.cpp` 约 `:104-116`），不受 can_be_optimized/skippable 标志驱动，`alias_analysis` 无对应别名边。DISABLED 用例待写 `DISABLED_selective_ssm_empty_seq_alias_edge`。
- **F28**（已由既有机制覆盖，2026-11-23 P1-PORT Step 7 更正）：oneDNN sum 的 dst 在依赖端口 k>0 时，`frozen_sum_downgrade_reason`（`program_helpers.cpp` 约 `:518-520`，F5 原因 4）已降级，由单测 `P1PORT_1_sum_dependency_at_nonzero_port_is_not_in_place` 锁定；冻结前的 `guard_superset` 偏保守。无需 DISABLED 用例。
- **MO-i**（候选，2026-11-23 P1-PORT Step 7 发现，归 MO 问题簇，只登记）：静态 in-place concat 接受多输出前驱。`concatenation.cpp` 约 `:162` 把前驱所有端口的 `_outputs` 换成 concat 的单端口 vector，动态路径 `primitive_inst.cpp` 约 `:1126` 只绑端口 0，`prepare_buffer_fusing` 的 `concat_in_place_optimization::match`（约 `:136-200`）不检查前驱端口与输出数。该配置运行期已坏。P1-PORT 的 C2 守卫（`get_add_fusing_type` exact 模式，路径上有多输出且被 concat 占用的节点则拒绝 sum）为其兜底。DISABLED 用例 `DISABLED_multi_output_predecessor_in_place_concat`（空壳）。
- **P1-PORT 后续（P1-B-D / P1-C 之前须处理）**：以下调用方仍按端口 0 回溯，须改为端口感知或写明端口 0 足够：`class_level_memory_dependencies.cpp` 约 `:113-116`（L1-12 叶子收集）与约 `:397`（`check_b4_prime`，仅 Debug）；`alias_analysis.cpp` 约 `:311`（kv_cache owner）；`program_helpers.cpp` 约 `:448`（前向链）与约 `:568`（F17 M1）；`basic_memory_dependencies.cpp` 的旧 root 回溯（P1-C 日落）。F27 与 MO-i 的 DISABLED 用例目前为空壳（alias_analysis_test.cpp），排期时补成真正刻画。F11 的 `start_port` 无区分性单测（需多输出视图 dynamic_quantize/nms_gather 喂 FC sum，融合不形成 sum），调度官记为未覆盖。
- **P1-PORT 语义说明（2026-11-23）**：端口感知回溯在 concat 之外比 HEAD 更宽松（多输出视图被读端口 k>0、nms_gather k>0 时 owner 为视图本身或对应输入 k）。这与运行期语义一致（dynamic_quantize 仅输出 0 重指；nms_gather、moe_mask_gen_reshape 为 k→k），`view_source_of_port` 是各端口别名输入的唯一来源。Release 仅做了 `-fsyntax-only -DNDEBUG -Werror`，完整 Release 构建（C10）与 P1-B-S 合入门槛一并欠账。
- **ENV-LRN**（环境/设备问题，2026-11-24，只登记）：`lrn_fp32_gpu.basic3` 自身通过，但其后任何触碰设备的用例会在 SetUp 报 `clEnqueueMemFillINTEL CL_OUT_OF_RESOURCES`，同进程内级联到块尾，新进程即复位。A/B 已证明不依赖 P1-PORT（HEAD 0da8eb7b11 去掉 P1-PORT 六文件的二进制同样复现）；2026-11-23 之前的 p1bs/baseline 全量日志里同一顺序曾通过，原因未查（疑设备/驱动状态）。复现：`ov_gpu_unit_tests --gtest_filter='lrn_fp32_gpu.basic3:axis_5/cum_sum_gpu_int32.basic/0'`。后果：全量日志在 lrn 之后可能出现整块级联失败，须以分块脚本（`run_full_unit_chunked.sh`）加新进程重跑失败项分诊，不作回归证据。分块脚本已知缺陷：用例名含 `-`（如 `KeepTopK=-1`）时过滤匹配失败，会被误记为 not_run，须把 `-` 换成 `?`。
- **P1-B-D 设计定稿**（2026-11-24，Step 2/3/7 设计模式已过，有条件放行后已并入 A1–A9）：`/home/ceciliapeng/repos/reports/memman/layer1/12_p1bd_design.zh-CN.md`。用户决定：设计先定稿、**实现等 P1-B-S 的 Step 8 前置恢复**（C2、C10、S2/S3、Qwen3-8B 无退化、compile_model 耗时均被用户暂缓）；第一切片接受非输出 MAY_VIEW 成员；块格式（`block_sizes()` 非空）一律拒绝；不先修 F24；不升版本号，前提是 P1-B-S 与 P1-B-D 同一发布版本交付（否则 3→4），Step 8 须禁用模型缓存。子开关 `enable_dynamic_concat_holder_pool`（须与 `enable_concat_holder_pool` 同开）。关键机制：`rebind_concat_slots` 挂在 `realloc_if_needed`（覆盖 holder 每次换块），SKIP 前驱陈旧指针须改指 holder 当前块并计数清零，每轮 padding/块格式闸门，L1-4 是第四处阻塞点。待用户数据：MiniMax-H3 部署队列类型（OOOQ 则收益为 0）、其 Slice→Concat 前驱是否为普通计算节点。未进入 Step 5。
- **MiniMax-H3 的 Pattern B 实况**（2026-11-24，Haiku 读 `/home/ceciliapeng/MX/openvino.pipeline.mx/src/cpp/src/modeling/models/minimax_h3` 所得，用户确认队列为 in_order）：主路径（`modeling_minimax_h3_transformer.cpp` 约 `:205-219` `for_each_packed_segment`、`:654-675`）为 4 段动态 `v8::Slice`（axis 1、边界为运行期 ShapeOf/Gather 计算）→ **Multiply/Add 计算节点** → 4 输入 axis-1 `Concat`，每块 4 次、默认 50 块共约 200 个 Concat；Concat 输出再被 4 个动态 Slice 读取（MAY_VIEW 读者）。所有路径 f32，3D 序列 (1, seq, hidden)，无 StridedSlice/VariadicSplit、无 KV/状态。t2va 时条件视频段长度为 0（`:202-204`），**零长度前驱是真实运行期情形**，对应设计的 SKIP 前驱陈旧指针用例。Concat 的前驱是普通计算节点，不会被 fb_role_conflict 拒绝（前驱本身若被融合为可跳过则另当别论，待实测）。AdaLN 部分（约 `:1014-1038`，快照）是 Slice→Concat 直连（静态行 Slice → axis-1 Concat）；**用户更正（2026-11-24）：实际部署中 AdaLN 可能被拆开，也可能与后面的 transformer 合并为一个模型，所以 Concat 输出不一定是 Result，shape 可能静态也可能动态，代码只是快照**。若合并，则为"静态 holder + 动态读者"的混合类，P1-B-S（L1-5 拒绝动态读者）与 P1-B-D 第一切片（rj_mixed_dynamic）**都会拒绝**，形成覆盖缺口，须在 P1-B-D 之后单独设计（候选 P1-B-X：静态 holder 带动态读者）；先要核实该合并形态的 exec graph。此前"MiniMax-H3 前驱类型"待用户数据一项就此关闭；文档中"200 个 Slice 对应 9 个 mem_ptr"与图结构（200 为 Concat 数）不符，以代码为准。
- **P1-B-D 实现状态**（2026-11-24）：本地提交 `1d280fca62`（基线 4dc84e213b，**不合入、不进 Step 8**；用户批准"实现、前置 C2/C10/S2/S3/Qwen3-8B/compile_model 耗时继续暂缓"）。Step 6 无阻塞缺陷（M1/M2 为仅 Release 的防护缺口）；Step 7 有条件放行，**B1 为合入阻塞**：前一融合轮唯一非空的前驱 padding 为 0，下一轮 oneDNN 64B 对齐检查使 concat 运行期被拒且 holder 同时换块时，该前驱不置 SHAPE_CHANGED、不 realloc，仍按偏移 0 写旧块，rebind 改指后 concat 读到未写内容（Release 静默错误）。**用户批准的设计修订（2026-11-24）**：在 `do_runtime_in_place_concat` 被拒分支（任何前驱入队前），仅当 concat 是已提升的 holder 时，对输出与 holder 同块的前驱 `clear_output_memory()` 并置标志，使其用自己的 DIRECT 缓冲；rebind 非融合分支对非 SKIP 且计数>0 的前驱改为清除而非改指；第二个本地提交承载此修复。合入前条件：C2（开关关闭配置下与 HEAD 二进制的分配 trace 对照）、C10、Qwen3-8B 无退化、禁用模型缓存、真实 maybe-CPU 动态读者用例（`force_implementations` 把读者强制为 cpu）、no-op 形状的动态 strided_slice 前驱用例。
- **P1BD-O3 / O4**（LEGACY 既有缺陷，2026-11-24 调查确认，用户决定只登记不修；开关关闭配置下同样错误，HEAD 即存在）：运行期决定按 layout（含 `padding::operator==`，只比尺寸与 mask）相等判断，而 layout 不含"前驱绑定在哪块内存"。**O3 镜像状态**：concat 运行期被拒（如 oneDNN 64B 对齐）后下一轮又融合，且 layout/padding 相等时，前驱不置 SHAPE_CHANGED/IMPL_CHANGED，仍写自己旧的 DIRECT 缓冲而 concat 读 holder 块，输出从下标 0 起错误；轮次 (0,16),(9,4),(0,4) 或 (0,4),(9,4),(0,4)。**O4**：被拒且 holder 不换块时，前驱仍别名 holder 块，concat 一个 kernel 内对同一块读 [0,b) 又写 [0,9+b)（读写重叠）；轮次 (0,4),(9,4),(4,4),(0,4),(9,4)，第 5 轮错于下标 144/208。DISABLED 用例 `DISABLED_legacy_rejected_then_fused_predecessor_keeps_own_buffer` 与 `DISABLED_legacy_rejected_runtime_concat_predecessor_aliases_holder_block`（轮次表在注释）。MiniMax-H3 的 OCL eltwise 前驱不受 oneDNN 64B 规则约束，基本不触发。**P1-B-D 开关打开时已由本地提交 `b08672dc74`（B1：被拒分支对与 holder 同块的非空前驱强制自分配）与 `c5d8fc70d7`（接受分支对未绑定到 holder 的非空前驱重新绑定，A4 在所有构建里对"已入队的非 SKIP 前驱将被改指"抛异常）修复**；A4 抛异常的生产风险待 Step 6/7 复核。
- **P1BD-O5**（候选，2026-11-24 P1-B-D Step 7，只登记）：concat 前驱在运行期 `is_valid_fusion()` 为假时，在重新获取内存前返回（`primitive_inst.cpp` 约 `:2873-2876`），execute 用未融合子图的输出替换 `_outputs[0]`；融合轮里其槽位不被写入，且该路径不置 `_enqueued_this_round`，A4 看不到。HEAD 的 LEGACY 已有，P1-B-D 将陈旧字节从 holder 旧数据变成其他池用户的数据。DISABLED 用例待写（eltwise 前驱带运行期无效的融合算子）。
- **P1-B-D Step 3.5 冲突裁决（2026-11-24）**：Step 6 称动态 FC 可作首个前驱并因 `_update_shape_done_by_other`（约 `:1366-1386`）跳过 `do_runtime_in_place_concat`；Step 7 称 FC/gemm 不是允许的 concat 前驱。调度官核对 `prepare_buffer_fusing.cpp` 约 `:71-77` 的 `available_pred`，不含 `fully_connected`，故 FC 前提被驳；但其他允许类型能否预置该标志仍在第四个提交中核实，加固保持。设计文档附录 A 记录 I-B5D′ 终稿。
- **P1-B-X**（候选，2026-11-24 用户决定登记、不并入 P1-B-D）：AdaLN 与 transformer 合并部署时的"静态 holder + 动态读者"混合类，被 P1-B-S L1-5 与 P1-B-D `rj_mixed_dynamic` 同时拒绝。须先核实合并形态 exec graph 与这些 Concat 的字节数（AdaLN 的 Concat 张量很小，约 (1, 4×hidden)，收益可能有限，属推断）；关键设计问题：静态块永不释放而动态读者寿命每轮可变，MAY envelope 能否覆盖。
- **MiniMax-H3 补充事实**（P1-B-D Step 7）：GPU 管线 `SliceToStridedSlice` 把 `v8::Slice` 转成 strided_slice（`common_optimizations.cpp` 约 `:139`），只有 no-op 形状才被标 cbo+rs，故"Slice 都是视图"不准确，多数是计算节点；动态 strided_slice 的 begin/end 在 prim 中为空向量，`all_zeroes(空)` 为真，可能被标成 rs（`mark_runtime_skippable_nodes.cpp` 约 `:122` 的 concat 守卫只作用于静态节点）。`primitive_inst.cpp` 中 P1-B-D 断言用 `#ifndef NDEBUG` 而非设计所写的 `P1B_DEBUG_CHECKS`，登记偏差。
- **P1BD-O1**（候选，可达性未核实，只登记）：concat 输出 shape 不变时 `do_runtime_in_place_concat` 在 `primitive_inst.cpp` 约 `:2358` 提前返回；若各前驱长度此增彼减、总长不变，槽位 padding 不会更新。DISABLED 用例 `DISABLED_dynamic_concat_compensating_pred_lengths_keep_stale_paddings`。
- **P1BD-O2**（显存浪费，非数值缺陷，只登记）：运行期从不融合的动态 concat，其前驱永久保持 `can_share_buffer(false)` 并直配（`prepare_buffer_fusing.cpp` 约 `:276`）。DISABLED 用例 `DISABLED_never_fused_dynamic_concat_preds_stay_off_pool`。
- **命名更正**：`11_p1_design_r2` §5 把 C-P2-1 标为"F24"，与本文件 F24（runtime-skippable 的 concat 守卫问题）冲突；以本文件为准，11 §5 的"F24"读作 C-P2-1。
- **F20**（候选，2026-11-18 F17 复审发现，复现后立项）：运行期 static impls 缓存（`kernel_impl_params::hash/operator==`）不含 onednn post-op 结构与 attrs；同参数节点跨节点命中时可能 post-op 类型/槽位错位（mul↔mul_inplace、sum↔binary_add、数据相关的 bin+eltw 合并），导致实参错绑或数值错误。F17 以 G-impl 只降不升兜底，sum 侧归 F5。
- **F17**（设计 r2 + 复审条件已定稿：`/home/ceciliapeng/repos/reports/memman/layer1/03_pre0_f17_design.zh-CN.md`，F19 提交后实施）：`can_use_mul_inplace` 接受单用户 input_layout，且不检查所有者与输出；动态 realloc 重绑时会把用户输入当作 FC 的 dst 写坏。修复复用 F6 的 helper。
- **F5 结构性部分**：单点重冻结、检测点之后调用点的收口、输出标记分歧时降级、导入时不覆盖。
- **F9**：统一 PO，内容如下：
  - 在 `post_optimize_graph` 末尾把动态 concat 前驱的移动写进 PO，使 `_exec_order` 等于过滤后的 PO。
  - network 侧保留移动分支，兼容旧模型缓存；该分支对已移动的 PO 幂等。
  - Debug 下断言两者逐项相同、无重复项。
  - 前驱有多个用户时取消该 concat 优化。
  - 同时修三项，各自先复现：前驱可能被执行两次；"shape 已由他人更新"标志残留；skipped_branch 在动态网络下按旧 PO 计算。
- **F4**：sum 写入视图，而视图所有者另有读者（`inplace_alias_conflict_test.cpp` 中的 4 个 `DISABLED_row11_*`）。在 Phase 1a 中实施。
- **缺陷 B**：sum 链 root 回溯遇到已优化 Concat 时，会走到第一个前驱。现状下对分配中性（pred(0) 已为 can_share_buffer(false)），P1-B 之后才有害。在 0.5a 中过渡修复：新旧两种回溯结果都标记，新标记用 `resolve_buffer_owner`；P1-C 删除旧标记。
- **F18**（2026-11-19 核实，主路径不成立）：自身 uid 在自身 restriction 中（调用点 `program.cpp:820-822`，门控 `program_node.cpp:231-235`）。只写锁定单测；`may_use_mempool()==false` 的残留常量节点以空 restriction 入池属未核实探针。
- **P1 本轮登记（2026-11-19；为避免与 F 序列撞号，使用前缀编号）**：
  - **MO 问题簇**（暂缓，用户决定）：输出标记按节点而非端口。子项 MO-a～MO-h，其中 MO-g = 只有端口 k>0 接 Result、且被 extract 时网络返回端口 0 的数据（正确性）；MO-f / MO-f2 = 动态多输出输出节点所有端口别名同一 ext_block、forward probe 不看端口。复现方案为用单测中的多输出 primitive 充当伪自定义算子。记录见 `10_multi_output_port_output_issue_record.zh-CN.md`。
  - **P1-Q6**（候选，2026-11-20 P1-B-S 实现探测）：`may_use_mempool()==false` 的 constant 非 data 节点以空 restriction 入池。结论：主网络中不可达（`propagate_constants` 把它们替换为 data 或删除，仅输出保留且走 DIRECT）；只写 DISABLED 用例 `DISABLED_constant_non_data_node_pools_with_empty_restriction`。
  - **P1-L1**（泄漏，归 P1-B.0）：`allocate_output` 绕池分支不释放 `curr_memory`。动态 holder 被拒（走池）后翻到融合扩容（绕池）时，池记录中残留 holder 的 uid。
  - **P1-L2**（泄漏，归 P1-B.0）：forward probe（约 `primitive_inst.cpp:1202`）用 ext_block 替换 `_outputs[0]` 时，不释放原池块。
  - **C-P2-1**（P2 强制约束，不是缺陷）：输出 reorder 被设为 cbo + requires_reinterpret 而不删除时，网络输出落在生产者的池块上。`past_outputs` 只登记 reorder 的 uid，现状靠 oooq 闭包与递归保护。P2 中含 USER_OUTPUT 成员的类，要么 death=∞，要么保留 LEGACY 的 oooq restriction。须加锁定单测。
- **候选，复现后再立项**：
  - F15：kv_cache 是网络输出时，`can_optimize` 跳过了 `is_output` 检查；
  - F16：permute 被标为视图后，它的 reorder 搭档被删掉；
  - CPU read_value impl 把 zp 拷到了 `output_memory(1)`，应为 2（约 `impls/cpu/read_value.cpp:87`）；
  - F7 未覆盖的情形：视图 reorder 作为带 padding 的输入、fsv16、动态。
- **归入后续阶段**：
  - mutable_data 传播的输出标记归 Phase 0.5 的 R5；
  - 无用户的输出 read_value 与独立 assign 之间缺少序边，归 Phase 1a；
  - 用 `war_safe` 放宽 F10，归 Phase 1a；
  - 统计 CNN 集（YOLO/CSP/UNet）的 sum post-op 数，以衡量 F10 的过度拒绝，在 Step 8 做。

### B0/D0 统计代码

**第一层只加统计代码，不跑目标模型**。验证以单元测试为主。目前只有 Qwen3-8B 有运行环境（`/home/ceciliapeng/openvino/run.sh`），其他模型的数据等环境就绪后再采集。

- **基础**：
  - B0/B1 基线：只构建，用于对比；
  - 结构化分配追踪钩子：在 GPU_DEBUG 下记录每次分配的节点、端口、字节数、是否走池、绕池原因。
- **第一层必需的统计**，现在实现：
  - 池外分配分类：按 in-place holder、sum root、多输出端口、网络输出、internal buffer、shape_of、kv/变量、其他分类，统计字节数与次数；
  - 多输出节点数，以及 idx>0 输出的字节数；
  - 网络输出字节占峰值的比例；
  - 同一 uid 共块的次数（F18 锁定探针）；池中幽灵用户（P1-L1/P1-L2）记录数；
  - P2 影子模式下"区间与 oooq 闭包"的逐对差集。
- **D 路线的统计**，随 D 路线实现，内容不变：
  - 切点数，区分 `update_shape_sync` 与 gather 回读；
  - SHAPE_CHANGED 的节点数与类型分布；
  - 异步编译过渡期内的节点数与持续轮数；
  - `slice` 类节点每轮的重算；
  - skip/in-place 决策的翻转次数；
  - 活跃集下界 L 与现状持有量。
- **验收**：每个统计项都有合成图单测，断言计数与已知构造一致。例如：in-place concat 整组计入 holder 类；sum root 计入 sum root 类；多输出节点的 idx>0 计入端口类。另外用 `run.sh` 在 Qwen3-8B 上跑通一次，确认统计输出可读。
- **最小子集（P1-B-S 的前置，用户 2026-11-19 选定）**：GPU_DEBUG 分配追踪钩子，加池外字节与次数按类别汇总，并配合成图单测。其余统计项随后补齐。
- **P1 子项顺序**：P1-B.0 → P1-B-S → P1-B-D → P1-C；P1-M 暂缓（MO 记录）；P1-O 并入 N2。有模型数据后再按池外字节占比调整。
- **目标模型**（环境就绪后采集）：
  - LLM：稠密 LLM（SDPA 与 PA 两条路径）、MoE、VLM；
  - 非 LLM：FlashVSR、SeedVR2（分辨率与帧数动态）、minimaxh3。
  - 需用户提供：输入 shape 分布、部署后端、多实例配置。

### Phase 0.5 `alias_analysis` + 影子校验

- **0.5a（已批准，设计见 11 §2 与 §7.3）**：
  - 接口：`tensor_id{uid, space{OUTPUT, INTERNAL}, index}`；7 值 `alias_kind`（含 MAY_INPLACE）；`offset_tag`；`fallback_reasons` 与 `domain_reasons` 两组位掩码，派生 `placement()`；`decision_source` 抽象。
  - owner 解析复用 `resolve_buffer_owner`，按取用边方向，禁止沿 cbo 回溯。
  - 不公开 PO 索引（只有私有的 `debug_po_rank`）。区间时间轴只能用 F9 的 exec order 纯函数，归 P2。
  - 行为中立，由 Debug 断言检查（只被新 owner 标记的节点必须本来就绕池）。
- **0.5b**：以下为原 Phase 0.5 的其余内容。
- 先由 Opus 产出独立设计稿，包括：R1 完整矩阵、R4、`war_safe` 的签名与三值语义、OPAQUE 别名清单、R5 补 `is_output`、V 规则命中后的处置。
- 角色类规则（R1/R2/V）只验证，修复留在声明点；时序类规则（R3/R4/R5）在单一检测点检测并降级。
- 本阶段不改行为。Debug 下默认开启，Release 需显式开关；oracle 在第一次 `execute()` 结束时采样。
- **持久类（kv/变量）设计输入（2026-11-20，0.5a 实施后核实，坐标为快照）**：
  - 现状：0.5a 中 kv/变量类整体落 LEGACY，但这是两条保守规则的副作用，不是"持久类"概念的结果。K2：压缩 kv_cache 与多输出 stateless_kv 全端口 opaque；K4：动态 kv 类含 read_value，而 read_value 只有 CPU impl，被判 maybe-CPU。`dm_persistent` 与 `STATE` placement 已存在，但 CPU 规则放宽前不可达。
  - `war_safe` 须显式处理，不得依赖回退：STATE 类与 CPU-fallback 的 kv 类都视为"只允许已登记持久写者的外部根"，或返回 Unknown。
  - 须新增"持久写者登记"（现无数据结构）：variable 别名的 kv_cache 输出 0、stateless_kv 的 past 端口、assign。
  - 持久类型被故意排除出 `dm_external_root`（`alias_analysis.cpp` 约 `:464-466`），尽管在 `is_external_buffer_root` 列表中。外部根判定须改为"外部根 ∨ 持久"或另设谓词，不能只读 `dm_external_root`。
  - 端口缺口：非压缩 kv_cache 的 beam_table（端口 1）无别名边也无 opaque，是单成员 `dm_persistent` 类，且内存由 impl 自行分配、不是 variable 内存。须与 STATE 语义对齐，并补测端口 1/2/3；核实 `get_outputs_count` 对压缩 kv_cache 是否报告 4 个输出。
  - 运行期 `do_runtime_in_place_kv_cache` 每轮可翻转，只省去"拷回 variable"，concat 等 kernel 照常执行；`war_safe` 与 D 路线按"总是原位"计入。

### Phase 1a `war_safe` + `resolve_inplace_conflicts`（in_order 静态）

接入 F4、F5 的 WAR 部分、F6、F8，新增 binary_mul_inplace 否决标志。前置：F9。

### P1 入池资格（在现有 memory_pool 上）

**共同机制：类级身份与 restriction 生成器**，只实现一次，P2 与 Phase 4s 都复用它。
- 由 `alias_analysis` 给出别名类，每个类有唯一的"池用户" root。只有 root 以 root uid 调用池，成员不得以 root uid 执行 get 或 release（`memory_set` 按 uid 去重、按 uid 删除）。
- **成员 uid 保留，并追加 root uid**（不替换；2026-11-19 裁决）。原因是：动态 envelope 类中可能分配的成员（某轮未融合的前驱）会以自身 uid 自行分配。
- 生成器三层接口：
  - L1 `pool_admission(class)`；
  - L2 `class_conflict_set(K)`：P1 为全部成员 restriction 的并集，P2 换成区间；
  - L3 `materialize_uid_sets()`：(2) R(root) 并入全部成员的 R(m)；(3') 对每个 R 中含成员的 X，双向登记 root；(4) 对称化。I-B1 双向完整，有证明。
- 统一路由 `allocation_route(node, tensor_id) → {POOL_AS_ROOT, ZERO_ALLOC_MEMBER, DIRECT, LEGACY_POOL}`，供 `allocate_output`、`do_allocate_memory`、估算函数与 B0/D0 统计共用。路由**只读序列化的标志**；`alias_pool_root_uid` 默认取哨兵值。
- 本层判定"两个类是否冲突"仍沿用现状的闭包规则；P2 再把这一判定换成区间。
- 运行在 `prepare_memory_dependencies` 末尾，即 F5 单一检测点之后。静态网络与动态网络都适用：动态网络按 MAY envelope 计入可能的 skip/in-place。
- 排除项沿用 Step 3 的回退规则。

**子项**：
- **P1-M 多输出节点**：降为 MO 问题簇记录，暂缓（用户 2026-11-19 决定）。端口级输出的完整支持（含 network 与插件按 (id, port) 绑定）建议与 N2 一起设计；`(uid, port)` 池粒度（P1-M2）并入 P2。
- **P1-B Concat holder 入池**（Crop 不需要：其 holder 是生产者，已入池）：
  - P1-B.0：修 P1-L1、P1-L2，无开关。
  - P1-B-S：静态 holder 作为类 root 走池；两用户前驱与级联内层 holder 不分配。开关 `enable_concat_holder_pool` 默认关闭；新增序列化字段 `pool_class_root` 与 `alias_pool_root_uid`，并提升版本号。
    - **状态（2026-11-20）**：设计 r4+r5（Step 2/3/7 已过，有条件放行）；已实现为本地独立提交 400ec9d4c7（基线 faa05fb295，**不合入、不进 Step 8**；用户批准"可实现不合入"）。仅处理静态 holder 且读者全静态（L1-5）；动态 holder/动态读者整类不提升，归 P1-B-D。**Pattern B 的真实目标模型是 MiniMax-H3（动态网络），P1-B-S 对其预期收益约 0**；P1-B-S 的价值是 P1-B-D 的前置基础设施与合成图 A/B（仅 holder 后继张量复用时总字节下降，其余仅池外分类与分配次数变化）。SeedVR2 480p（2026-10-10 运行）全静态、33 个静态 in-place holder 约 7.38GiB 仅为旁证，部署形态下是否静态未核实。零分配仅限无视图/rs/cbo 其他用户的 MUST_SUBREGION 成员；allocation_route 只让 POOL_AS_ROOT/ZERO_ALLOC_MEMBER 读序列化标志，DIRECT/LEGACY_POOL 保留运行期 `impl_params.can_be_optimized()` 谓词；单用户 reorder/permute 前驱当前也会瞬时直配（reorder.cpp:233、permute.cpp:121 绕过 do_allocate_memory）。runtime_requirements_version 2→3。开关不进缓存键：Step 8 须禁用模型缓存或把开关纳入缓存键。network.cpp 有一段仅 Debug 的只读断言（对 11 §3.7 的登记偏差）。合入前必须完成：用户手动 C2（先在 faa05fb295 上、再在 P1-B-S 开关强开）、C10 Release 无 debug caps 编译、S2/S3、Qwen3-8B 对照、compile_model 耗时。
  - P1-B-D：动态 holder 入池，新增 `rebind_concat_slots`（`bind_class_members` 的雏形）兜底；holder 须无 padding；有独立子开关。
  - 前置：缺陷 B 已过渡修复（0.5a）。
- **P1-C sum 链首 root 入池**：root 作为类 root，类覆盖链上全部成员及其全部用户，不再 `can_share_buffer(false)`。
  - 前置：F5 全部完成、F6、F10、F11、Phase 1a、缺陷 B 已修。
  - stateless_kv 与 kv_cache 属于 kv/变量，不在本项内。
- **P1-O 网络输出的前段复用**〔候选，须 Opus 先确认语义〕：
  - 思路：输出缓冲在其产出节点执行之前处于空闲状态，可以允许寿命在产出点之前结束的中间张量借用它。
  - 前提有两条，任一不满足就不做：
    1. 用户不会在下一次推理执行期间继续读取上一轮的输出。要核实异步推理、GenAI 流水线，以及用户用 `set_output` 提供 buffer 的情形。
    2. 输出缓冲的跨轮持有方式不变。
  - 收益看 B0/D0 统计的输出字节占比，例如视频模型的帧输出。

**Go/No-Go**：
- **单测层，现在可执行**：
  - 模拟 DiT 残差块、窗口 Concat 和多输出节点的合成图上，池外字节按类别下降；
  - 开关打开与关闭时输出逐元素一致；
  - OOOQ 用例比特级不变；
  - Qwen3-8B 无退化。
- **模型层，环境就绪后补测**：
  - SeedVR2 DiT whole：ETC 大幅下降（P1-C 估算约 14GB，7.6GB 窗口 Concat 为 SeedVR2 事实，但 Pattern B 的出处与目标模型是 MiniMax-H3（动态，归 P1-B-D），均以实测为准），峰值同步下降，输出逐元素一致；
  - FlashVSR、minimaxh3：池外字节按类别下降；
  - CNN 回归集（VAE、UNet、ResNet、YOLO）：峰值不升，kernel 数不变；
  - OOOQ：比特级无变化。

### P2 区间冲突规则（in_order，静态与动态）

**做法：在池内直接比较区间**，不把区间展开成 restriction 集。展开成集合最坏需要 O(N²) 的存储，本质上仍是两两冲突表。

- **接口**：
  - `memory_user` 增加类区间 `[birth, death]`，以 `_exec_order` 序号表示，并记录所属的类 root；
  - `memory_pool::get_memory` 及 `get_from_non_padded_pool`/`get_from_padded_pool` 增加区间参数。
- **冲突判定**：`has_conflict` = 现有 restriction 命中 ∨ 与块上任一用户的区间相交 ∨ 与块上任一用户属于同一个类（含同一 uid，承接 F18）。
- **何时只用区间**：只有 in_order 且双方区间都已知时，才单独依据区间判定。区间未知的用户按 `[0, ∞)` 处理，即与任何用户都冲突。
- **restriction 集的变化**：
  - 对双方都有区间的节点对，不再由 oooq 闭包生成 restriction；
  - 仍保留的部分：OOOQ 网络的全部 restriction、回退节点（CPU、shape_of、kv/变量）相关的 restriction、`past_outputs`；
  - 与 P1 的类级身份一样，restriction 中保留成员 uid 并追加类 root。
  - **C-P2-1**：含 USER_OUTPUT 成员的类（包括输出 reorder 为 cbo 视图时的生产者类），要么 death=∞，要么保留 LEGACY 的 oooq restriction。
  - 区间时间轴只能用 F9 的 exec order 纯函数（与 `build_exec_order` 共用）。
- **区间的来源**：由 `alias_analysis` 产出，一次取快照。birth 是类 root 的序号，death 是类内全部成员的全部用户序号的最大值（跳过 shape_of）。internal buffer 的区间为 `[n, n]`。
- **动态网络**：`_exec_order` 在构造期冻结，skip/in-place 的翻转已由 MAY envelope 覆盖，所以运行期重分配也带同一份区间。
- **`may_alias`**：in_order 路径改为比较两个张量的类区间，以及它们是否落在同一块上；之后 in_order 不再运行 oooq 闭包。OOOQ 保持现状。
- **为第二层做准备**：块上已按区间记录占用，Phase 4s 只需把"整块复用"换成"块内 offset 装箱"。
- **前置**：F9；Phase 0.5 影子校验干净；P1 的类级身份。
- **影子模式**：先只计算、不生效，与 oooq 闭包的结果逐对比较。区间判为冲突而闭包判为不冲突的节点对，视为缺陷，必须为零；反方向的差集逐项列出，即为收益来源。
- 单测：见 §2 Pattern A。
- Go/No-Go：
  - Pattern A 类模型的峰值下降；
  - 动态 LLM 只要求无退化（文档 10 显示 restriction 在 LLM 上只占约 0.5–1 MiB）；
  - OOOQ 比特级无变化。

### B0-ARENA 大块分配可行性实测（第二层，Phase 4s 之前）

执行者为 profiler，结论决定段容量 C：
1. 在 Arc 上读出 `max_alloc_mem_size` 与 `max_global_mem_size`。
2. 默认配置下分配一块略超上限的 `usm_device`，确认会抛异常。
3. 在大于 4GB 的父缓冲上切出偏移超过 4GB 的子缓冲，分别跑一个 OCL kernel 和一个 oneDNN primitive；比较带与不带 `-cl-intel-greater-than-4GB-buffer-required` 时的正确性与延迟。

默认设计为**分段 Arena**：每段不超过 C（C ≤ `max_alloc_mem_size`），张量不跨段，对齐取 `max(sub_buffer_base_alignment, 64B)`。只有第 3 项证明安全且无退化时，才允许 C 超过上限。

### Phase 0 `AliasSink`

- 覆盖全部声明点：prepare_buffer_fusing、handle_reshape、remove_redundant_reorders、select_preferred_formats、mark_runtime_skippable、构造期视图。
- 行为中立，附分配序列的二进制对比测试。
- 作为 Phase 4s 别名类的影子校验。

### Phase 4s 静态 Arena（更换池算法）

- **范围**：in_order 静态网络。
  - 直接复用 P1/P2 的别名类与类区间，只把"按块贪心取池"换成"带 offset 的离线装箱"。
  - 内部缓冲作为独立张量参与装箱，区间为 `[n, n]`。
  - 求解器按 B0 的真实图数据选定，静态与动态网络共用。
  - 按 B0-ARENA 的结论分段。
  - 计划不序列化，模型缓存导入后重算。
- **收益来源**：块内 offset 装箱，免去整块复用造成的浪费。文档 10 中这部分约 39 MiB；在静态网络上以实测为准。
- `may_alias` 对 Arena 张量改为查计划中的 offset 是否重叠。OOOQ 网络整网走 LEGACY。
- **前置**：P1、P2 已通过 Step 8；Phase 0 的影子校验干净。
- **审计**：必须给出文档 4 不变式 3、4、5、9 的具体机制，否则退回。
- **Go/No-Go**：
  - 目标模型的峰值在 P2 的基础上进一步下降，输出逐元素一致；
  - OOOQ 与动态网络比特级无变化。

### N2 输出域（排在 D6 之后，待设计）

- 用户可见输出单独放在一个输出域，与激活 Arena 分开（参考文档 7 的 ExternalBindings）。
  - 用户经 `set_output` 提供了 buffer：直接写入，不另行分配。
  - 没有提供：按轮分配，上一轮的输出在用户取走前不得被覆盖。
  - 输出在本轮内还被其他节点读取：区间延长到本轮末尾，但不占激活 Arena。
- 被视图别名的输出，整个别名类归入输出域。
- 若 P1-O 已落地，则 N2 只负责剩余部分。是否实施，取决于 B0/D0 统计的输出字节占比。

### 〔可选〕Phase 3a 去掉 sum 跳数上限

- **收益**：P1-C 之后，sum 链断开只多出一个 binary_add 输出张量，它照常入池复用。所以 3a 剩下的收益只有这个张量和一次原位写，在 DiT 上估计约 0.6GB。只有 P1-C 的 Step 8 显示剩余差额值得时才启动。
- **方案**：采用 M1 + M4。新谓词只在 in_order 静态网络且开关开启时生效，开关默认关闭；OOOQ 保持上限 7。两个谓词按队列类型并存，早期调用方也使用同一个按队列选择的谓词；CPU/shape_of 读者那一跳回退到有界结果。
- **风险**：
  - **R1 早期决策的连带损失**：在 F5 检测点之前，有三处已按 sum 做了不可回收的决定：`prepare_buffer_fusing` 拒绝 concat in-place、`layout_optimizer` 不融合输入 reorder、`remove_redundant_reorders` 保留 simple reorder。这三处同时也是一致性守卫，不能放宽。静态 CNN 与混合结构风险高（UNet/YOLO/CSP、fsv16↔bfyx 边界、SD UNet、SeedVR2 VAE）。
  - **R2 正确性**：去掉上限本身不新增直接用户的 WAR，风险集中在 F11、F10 与缺陷 B 所描述的情形，以及链上中间节点的其他用户。
  - **R3 判定一致性**：早期调用方、冻结点与 F5 共用一个结构谓词 S。`add_required_reorders` 与 `fuse_primitives_with_layout` 改图后可能导致升级，由 Phase 0.5 的 R1 兜底。
  - **R4 编译时间**：最坏 O(N²)，以 PO(target) 截止或对结果缓存来缓解。
  - **R5 测试与数值**：`ancestor_walk_depth_boundary` 改写为"in_order 深度 8/16 判 sum，OOOQ 保持 7/8 边界"；该测试文件中的静默 return 改为 `GTEST_SKIP`；做输出比对。
- **Go/No-Go**：
  - SeedVR2 DiT 的 ETC 在 P1-C 的基础上进一步下降，输出无差异；
  - CNN 集的 concat in-place 数、存活 reorder 数、kernel 数都不增加，否则延迟退化须 ≤ 1%，达不到就对该类模型关闭；
  - 编译时间增量 ≤ 5%，F5 降级数接近 0；
  - OOOQ 与动态网络比特级无变化。
- **单测**：
  - 深度 8/16/32 的残差链：in_order 判 sum，OOOQ 判 binary；
  - 视图的其他读者晚于 p：降级；
  - 早期判非 sum、之后变为 sum：不得升级；
  - 统计连带损失；
  - 输出 reorder 不被删除；
  - 动态节点恒为 not_supported；
  - CPU/shape_of 读者回退；
  - F5 降级后 attrs、`fused_prims_onednn` 与 impl 三者一致。

### 动态 Shape 路线（D 阶段）

**谱系**（出自文档 8 §8.3）：

| 尺寸可知的时机 | 同步点 = 0 | 同步点 > 0（检测、分割、MoE、控制流） |
|---|---|---|
| 逐节点（A1） | HWM 计划 + 中途溢出 | 同左，再加 DEFERRED（B1） |
| **先推导（A2）← 采用** | 整图推导 → 重放或重规划 → 入队 | **按段推导 → 在线续排 → 入队（B2）** |

A2 是只有一段的 B2，所以只实现 B2 一套机制。

**每轮流程（in_order）**：
1. **推导**：按 `_exec_order` 推导当前段。产出精确 layout，并确定 impl（`update_impl` 前移，按轮冻结，异步编译完成后的切换推迟到下一轮）、内部缓冲尺寸、本轮的 skip/in-place/kv_cache 原位决策，以及期望的权重 layout。推导期间不入队任何 kernel。
2. **取计划**：用轮签名查计划缓存。签名包括张量字节尺寸、别名决策、impl 选择与内部缓冲尺寸。默认要求签名完全一致才算命中，命中则重放；未命中则装箱并写回缓存。"每个张量都不大于原槽位且浪费不超过阈值"的放宽判定只能在开关后启用，且须按轮报告浪费量。
3. **绑定并入队**：按 offset 在 Arena 上切子缓冲，绑定后入队。
4. **切点**：执行已有的 `finish()` 与数据回读，然后推导下一段。跨段存活的张量保留 offset；新张量在当前空闲区间里在线续排，放不下时追加 chunk。本轮结束后，下一轮按观测到的尺寸合并成单个计划。

**规则**：
- **上界**：不依赖上界。每段入队前尺寸已精确可知，所以不会在执行中溢出；设备显存真不够时照常报 OOM。
- **容量**：计划峰值超过当前容量就换更大的 Arena；低于容量的 r 倍就缩容，r 可配置。旧 Arena 等上一轮执行完再释放。Arena 按 B0-ARENA 的结论分段。
- **别名类按轮划分**：用本轮的实际决策划分，不使用 MAY envelope，由同一个 `alias_analysis` 引擎产出。视图跟随 root 的 offset。
- **编译期启发式计划**：
  - `compile_model` 时用一个启发式初始 shape 推导并规划一次，只缓存计划，不做物理分配。首轮命中时按该计划一次性分配。
  - shape 来源优先级：用户配置的 shape profile > IR 中有界维度的取值 > 都没有则不生成。具体规则在 D6 中定。
  - 推导必须是纯计算，不得改写 layout、impl、kernel、padding。
  - 它不是上界，不得用来限制或撑大运行期 Arena。
  - 编译期只有 shape-agnostic impl，运行期换 impl 后签名会变，所以命中率以实测为准。
- **另外走 LEGACY 的部分**（在 Step 3 回退规则之外）：loop/condition 的 body 网络，外层把控制流节点当作切点；推导失败或签名中含 Unknown 的节点。计划不序列化。
- **切点**：`has_runtime_deps`，以及 `do_runtime_skip_gather` 的 `finish()` 加索引回读（约 `primitive_inst.cpp:2016-2018`）。
- **`update_weights`**：它在 prepare 阶段会入队 reorder（约 `:3139`）。所以推导阶段只确定期望的权重 layout，reorder 的入队留在执行阶段。
- **推导范围**：
  - 输入 layout 不变的节点在 `update_shape` 中比较完 layout 就提前返回（约 `:438-466`）。文档 10 中生成轮每步只有 36 个 PA 节点重算 layout。
  - 例外一：异步编译过渡期，shape-agnostic impl 每步都进入 `update_impl`。
  - 例外二：未覆写 `get_shape_infer_dependencies()` 的类型（如 `slice`），输入来自 GPU 时每步都重算并 `finish()`。

| 阶段 | 内容 | 门槛 |
|---|---|---|
| D0 | 见 B0/D0 | 数据齐全 |
| D1 | 从 `update_shape` 拆出纯函数 `infer_layouts()`，连同 skip/in-place/reorder/permute 的决策谓词；整图影子推导，并与实际结果逐节点比对 | 目标模型集上零差异 |
| D2 | 影子规划器：用 D1 的结果加 `alias_analysis` 的区间做每轮装箱；独立校验器检查 offset 重叠的张量区间不相交；报告计划峰值、L 与现状持有量 | 零违例，并给出装箱质量数据 |
| D3 | 决定与应用分开：`update_impl` 前移并按轮冻结；`update_weights` 拆为推导和执行两部分；concat/crop/reorder/permute 的决定与应用分开 | 输出与执行序列逐项不变 |
| D4 | A2 单段落地：只对切点为 0 的网络启用，开关默认关闭；复用 Phase 4s 的规划器与绑定，加上计划缓存与容量策略 | 输出逐元素一致，持有量低于现状 |
| D5 | B2 分段：跨段在线续排、追加 chunk、迭代边界合并 | 同 D4，覆盖有切点的模型 |
| D6 | 编译期启发式计划与缓存 | 给出命中率数据；未命中时与 D5 结果一致 |

**前置**：
- D2 依赖 Phase 0.5 与 F9。
- D4 依赖 Phase 4s、Phase 0.5 影子校验干净、F17。
- 动态节点的 sum 恒为 not_supported，所以 `war_safe` 不是 D4 的前置。
- **前向约束**：动态 realloc 只在 `IMPL_CHANGED` 或显式请求时，才会重绑 `get_reused_eltwmem_idx` 用户的 dst。P1-B/P1-C 入池、Phase 4s 与 D4/D5 的 Arena 绑定，只要改变了依赖内存，就必须同时重绑这些用户。

## 2. 单元测试强制要求

触达 Pattern A/B/C 判定或原位冲突处理的改动，必须在 `src/plugins/intel_gpu/tests/unit/` 下补齐合成图单测，不得以"现有测试已覆盖"为由省略。各条按本阶段已落地的组件适用，不适用的标 N/A 并说明原因。依赖 oneDNN 的用例必须用 `GTEST_SKIP`，不得静默 return；SKIP 计为"未覆盖"。

- **Pattern A**：
  - 构造 N 个共享上游、彼此结构不可达的兄弟分支。验收标准是可靠性：区间法判为不冲突的每一对，在 in_order 执行序下真实生存期不相交（以模拟执行作 oracle），并逐项列出与旧 `user_bitmap` 闭包的差集。
  - 覆盖"多输入汇聚"边界 A→X→B、A→u、B→u：不得发生别名穿透或生命周期漏判。
- **Pattern B**：
  - 区分 Concat（holder 是 concat 本身）与 Crop（holder 是生产者）。
  - 静态：融合组全部前驱的区间并入 holder 所在的类，holder 由池、Arena 或 `memory_manager::acquire()` 分配，而不是直接 `_engine.allocate_memory()`。
  - LEGACY 动态：`do_runtime_in_place_concat()` 按"融合总会发生"的最坏情况合并区间。
  - D 路线：按本轮实际决策划分别名类，不越界复用。
- **Pattern C**：
  - 深度超过 `max_ancestor_walk_depth` 的残差链。
  - 冻结值与实时值分歧，危险方向是"冻结为 sum、实时判非 sum"。须构造：mutable_data 在 sum 标记之后传播输出标记；post 阶段生成视图；`post_input_reorder` 使跳数加 1；模型缓存导出再导入。断言 impl 的 post-op、`fused_prims_onednn` 与 `share_buffer` 三者一致，分歧只能导致降级。
  - binary_mul_inplace 经 `get_reused_eltwmem_idx` 返回索引、而 `basic_memory_dependencies` 未标记的既有风险。
  - 守护动态节点恒为 `not_supported`（不需要 oneDNN 设备）。
  - 链 `{root → S_1 → … → S_k}`：root 所在类的区间为 `[birth(root), max_death(S_k)]`，并覆盖全部成员的全部用户。
- **原位冲突**：
  - sum 写入视图，而所有者另有后续读者；
  - Concat 前驱有第二读者，且 sum 写入该 concat；
  - sum 依赖是 input_layout 或其视图，之后被 `set_data` 替换；
  - 视图链上有网络输出；
  - 写者经另一端口读本类；
  - read_value 是网络输出；
  - reorder→permute 中 permute 是网络输出。

  每项断言输出与参考一致；修复后断言确实发生了降级或拒绝。
- **多输出节点**：P1-M 暂缓；MO 复现单测（R1–R4，见 `10_multi_output_port_output_issue_record.zh-CN.md` §5）择期补。P1-B 须补 TB17（成员 internal buffer 不与 holder 共块）、TB18（单向边下 I-B1 白盒断言）、TB19（两用户前驱的构造顺序）。
- **刚性排除**：节点对中至少一方是 CPU impl 或 `is_in_shape_of_subgraph()` 时，验证仍保留原有的保守隔离。

## 待核实（实施前须复现）

- 缓存导入后三份 sum 判定错配的表现。
- Rule 2 caveat 是否真的漏判。
- 恒等 reorder（含 padding）能否选到 impl。
- 清除动态 Concat/Crop 的可优化标志后，前驱的动态 padding 是否正确。
- LoRA 在 adapter 非空时是否原位写入。
- USM 释放是否阻塞。
- dynamic_quantize 的可跳过标志由谁翻转。
- `basic_memory_dependencies` 用 `eltw_dep != 0` 判断"已找到第一个 sum"，outer_dep 下标为 0 时会失效。
