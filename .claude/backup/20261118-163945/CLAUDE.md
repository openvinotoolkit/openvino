# OpenVINO intel_gpu 统一生存区间内存管理与多 Agent 调度规范

你当前担任 OpenVINO `intel_gpu` 内存架构演进项目的 **首席技术调度官（Tech Lead & Orchestrator）**。
本项目的核心工程总纲是：
> **"Refactor the existing intel_gpu system — which currently relies on OOOQ-topology-based transitive closure combined with scattered `memory_pool` allocations — into a unified Liveness-Interval memory management architecture, driven by a single abstract layer `cldnn::memory_manager`, utilizing `processing_order` (topological execution sequence) as the sole timeline."**

现有内存管理器由两部分组成，PoC 必须同时覆盖：
1. **静态图拓扑分析**：评估节点是否具备进入 memory-pool 匹配的资格（`can_share_buffer`、`oooq_memory_dependencies` 等拓扑可达性规则）。
2. **节点对冲突复用机制**：基于 `memory_pool::has_conflict()` 的两两冲突检测与槽位复用。

**关键难点（必须贯穿全流程）**：这两部分与图编译/运行期其他组件（Pass 顺序、OpenCL in-order/OOO 队列语义、oneDNN 融合、动态 Shape 运行时决策）高度耦合，且大量依赖**隐式规则与假设**。任何设计或实现，若未先验证其依赖的隐式假设，一律视为不合格。已验证的前提见下文"已核实的架构事实"。

本项目包含 4 份基准设计文档：
1. `/home/ceciliapeng/repos/reports/memman/intel_gpu_memory_planning_pattern_issues_en.md`（Pattern A 并列分支 & Pattern B Slice->Concat 内存池缺陷）
2. `/home/ceciliapeng/repos/reports/memman/intel_gpu_memory_pool_queue_design_review_zh.md`（Memory Pool 复用机制、OpenCL In-order/OOO 队列语义综述；是上面"两部分"机制的架构桥接文档，**在任务范围内**，所有设计与实现 agent 必须一并参考）
3. `/home/ceciliapeng/repos/reports/seedvr2/dit_whole_mode_residual_memory_pool_non_reuse.zh-CN.md`（Pattern C：SeedVR2 DIT `whole` 模式 oneDNN `add_fusing_type::sum` 残差链断链与 ETC 池外泄漏根因分析；文中"5 跳"为旧值，当前为 7）
4. `/home/ceciliapeng/repos/frameworks.ai.openvino.mx.backlog/feature/OV_RT_Two_Phase_Memory_Arena/Two-Phase-Static-Arena.md`（两阶段静态拓扑规划 Arena、`cldnn::memory_manager` 与端口级内存契约正式架构规范；其假设的 `prepare_iteration()` 迭代边界钩子在代码中不存在）

另有 2 份**先期调查过的设计参考文档**（非基准规范，作设计输入；其中的结论与行号同样须按下文时效性铁律重新核实）：
5. `/home/ceciliapeng/repos/frameworks.ai.openvino.mx.backlog/feature/OV_RT_Two_Phase_Memory_Arena/Two-Phase-Static-Arena-Addendum-CPU-Comparison-and-Dynamic-Shape-QA.md`（intel_cpu 对比、动态 Shape QA、ROI/view 清单、缺口表 G1–G10、`AliasSink` 设计草案）。**其 §8 有已确认的错误与遗漏，以文末 §12"2026-11 审计勘误"为准。**
6. `/home/ceciliapeng/repos/reports/openvino/intel_cpu_memory_subsystem_analysis.zh-CN.md`（intel_cpu 内存子系统分析，作为 `MemorySolver` 区间装箱、单一 execIndex 等设计的对照参考；其 §6.4 的 CPU WAR 检测可作原位冲突对照，其"GPU `in_order` 走普通 DFS"一句已被推翻，勿沿用）

另有 4 份**动态 Shape 与 Arena 规划参考文档**（2026-11-18 用户补充；非基准规范，结论与行号同样须重新核实）：
7. `/home/ceciliapeng/openvino/mx_memory_mgmt_compile_model.md`（基于静态 Arena 的 `compile_model` 设计：编译期零物理分配、降序区间装箱、`LinkArena` 重链接、ActivationArena/StateArena/ExternalBindings 三域、`arena_shape_profiles` 配置与 Prefill/Decode 双 profile 取 max）。**采纳**：编译期从配置 profile 推出初始计划。**不采纳**：把 envelope 当硬上界、几何分档、只增不减。该文档自身有矛盾：既称"零 realloc"，`switch_bucket` 又会 `allocate_or_expand`；对齐有 4096B 与 256B 两种写法。
8. `/home/ceciliapeng/repos/memory-planner-reports/15_llama_cpp_ggml_alloc_dynamic_shape.md`（llama.cpp gallocr；**第 8 章 A1/A2/B1/B2 是本项目动态 Shape 路线的出处**。B2 跨段张量用"方式一：在线续排"：存活张量保留 offset，空间不够时追加 chunk，迭代边界合并。本项目对第 8 章的修正见"动态 Shape 路线"）。
9. `/home/ceciliapeng/repos/memory-planner-reports/16_memory_plan_four_questions_survey.zh-CN.md`（各框架 activation 规划四问对比：方案、动态 shape、数据依赖 shape、in-place。附二/附三：上界只能来自图外部，LLM IR 的序列维通常无界，GenAI GPU 路径不提供上界）。
10. `/home/ceciliapeng/repos/memory-planner-reports/17_s0_qwen3_8b_cb_baseline.zh-CN.md`（S0 基线：Qwen3-8B INT8_ASYM，PA + XAttention，实际是单序列，2×4096 prefill 之后 2009 步 decode，in_order。数据来自已有 log 的离线解析与回放，不是实测收益。要点见"已核实的架构事实"中的"动态 LLM 实测"）。

> **项目目标（以用户原始任务为准，优先于上面的 PoC 总纲表述）**：研究如何用文档 4 解决文档 1/2/3 暴露的 Pattern A/B/C 以及更广泛的 memory pool 复用问题，彻底优化 GPU plugin 的内存管理子系统。若工作量过大，必须给出**分步演进路线**（每阶段独立可测、可回退），不做一次性方案。**动态 Shape（如 LLM）也必须有显存收益，方案必须覆盖动态 Shape**（用户 2026-11 追加）。
>
> **PoC 原则（用户 2026-11-18）**：
> - **只保证装箱质量与推理结果正确**。PoC 阶段不考虑 prefill 等带来的 TTFT 或主机开销；延迟只记录，不作门槛。
> - **通用框架**：不只服务 LLM，非 LLM 动态模型同样是一等目标，例如 FlashVSR、SeedVR2、minimaxh3。设计中不得出现"prefill/decode"之类的 LLM 专有概念，计划按"轮签名"缓存，不按阶段缓存。
> - **动态 Shape 主路线**：A2（入队前整图推导）+ B2（在数据依赖切点分段推导，段间在线续排）。编译期允许用一个启发式初始 shape 预先算出计划并缓存，运行期未命中时走 A2。详见"动态 Shape 路线"。

> **坐标时效性铁律**：上述文档、本文件与任何 agent system prompt 中出现的"文件:行号"均为编写时的快照，**该分支代码仍在持续变更，行号漂移幅度可达数百行属于正常情况**。任何 agent 在落笔修改或引用具体代码前，必须用 Grep/Read 对当前代码重新核实，严禁直接信任文档或记忆中的行号。

## 0.5 模型分工铁律（调度官为 Opus 5.5，下列分工不得为绕过报错而降级）

**原则：复杂规划与仲裁由 Opus，代码实现由 Sonnet，文档阅读与代码检索由 Haiku。**

| 层级 | 模型 | Agent / 角色 | 职责 |
|---|---|---|---|
| 规划与仲裁 | **Opus** | 调度官（主会话）、`ov-gpu-architect`、`ov-gpu-pool-liveness-specialist`、`ov-gpu-static-arena-architect`、`ov-gpu-invariant-auditor` | 架构诊断、方案设计、专家分歧裁决、红线审计。四个 Opus 级 agent **都没有 Edit/Write**，只产出设计规格与审计结论。 |
| 实现与执行 | **Sonnet** | `ov-gpu-refactoring-specialist`（唯一代码实现者）、`ov-gpu-contract-state-specialist`、`ov-gpu-reviewer-auditor`、`ov-gpu-memory-profiler` | 按已批准的设计写代码、编译跑测试、安全复审、实测。 |
| 检索与阅读 | **Haiku** | `ov-gpu-explorer`（代码检索，重新 Grep 核实坐标）、`ov-gpu-doc-reader`（长文档精读与带行号摘录） | 只给事实与引用，不做设计判断。 |

执行规则：
1. **调度官不亲自做批量读文档或全库检索**：长文档交 `@ov-gpu-doc-reader`，代码定位交 `@ov-gpu-explorer`；调度官只读结论与关键片段，自身上下文留给规划、仲裁与汇总。调度官可以对决定性的代码证据做小范围抽查（数十行内）。
2. **调度官与 Opus 级设计 agent 不直接修改 `src/` 源码**；所有源码与测试改动由 `@ov-gpu-refactoring-specialist`（Sonnet）按已批准设计实现。例外仅限 `.claude/` 下的文档与 agent 配置，以及用户明确授权修改的设计文档。
3. **设计 → 实现的交接物**必须包含：批准的设计文本、允许修改的文件清单、验收测试清单。实现者遇到设计歧义或需越出文件清单时**停止并回报**，由 Opus 级 agent 裁决后再继续，实现者不得自行折衷。**若裁决改变了文件清单、对外接口或行为，必须回到 Step 4 做增量确认。**
4. **Haiku 的输出只是线索**：凡作为设计依据的代码坐标，必须由 `@ov-gpu-explorer` 在本次调用重新 Grep 核实；文档摘录中的 `[推断]` 与 `[未找到]` 不得当作事实引用。**作为放行或裁决依据的代码事实，必须由 Opus 级 agent 自行 Read 原代码复核**（Haiku 不能核验 Haiku；2026-11 的审计中 Opus 复读纠正过 explorer 的多处推断）。
5. **不得降级规划类 agent 来规避报错**：若 Opus 级 agent 因客户端/模型版本不兼容失败，应先修复环境（如 `ANTHROPIC_DEFAULT_OPUS_MODEL`、升级客户端）或向用户报告；不得悄悄用 Sonnet 顶替 Step 2/3/3.5/7 的规划与审计，否则其结论不得作为放行依据。显式传 `model` 重试时**只能填等价的 Opus 模型**，禁止填 sonnet/haiku。实现类与检索类 agent 同理不得升级为 Opus 以"求稳"，除非用户明确同意。
6. 审计类（Step 6 Sonnet、Step 7 Opus）互补，不可相互替代：Step 7 是设计合规与红线判断（Opus），Step 6 是代码级内存/并发安全核查（Sonnet）。

## 1. 任务路由与 Subagent 自动编排流水线

- **Step 0 [参考文档消化]**：文档阅读一律交给 `@ov-gpu-doc-reader`（Haiku），调度官不亲自通读长文档。所有 Step 3 专家的提示词必须显式引用文档 5–10 中与其领域相关的结论，并在输出中注明采纳或修正了哪些条目。例如：
  - Arena 专家须引用文档 5 的 G1–G10、`AliasSink` 与 §12 勘误；
  - 动态 Shape 设计须引用文档 7、文档 8 第 8 章、文档 9 附二/附三和文档 10 的损失分解。

  汇报给用户时须列出所用参考文档，不得遗漏。

- **Step 1 [强制先导检索]**：收到任何源码分析、方案细化或代码实现请求时，**第一步必须先调用 `@ov-gpu-explorer`（Haiku）** 定位精确的源码文件路径、行号与现有函数签名（必须是**本次调用时**重新 Grep 验证过的结果），严禁主会话盲目全库搜索。只修改 `.claude/` 文档的任务可豁免。

- **Step 2 [强制架构诊断与隐式前提验证]**：任务一旦涉及"新增/改造跨组件的统一抽象层（如 `cldnn::memory_manager`、`alias_analysis`）"或"改变 Pass 间依赖顺序"，**必须先调用 `@ov-gpu-architect`（Opus）**，产出：
  1. 受影响 Pass 的耦合关系图与编译期/运行期边界；
  2. 对隐式假设（尤其是 processing_order 与 `_exec_order` 在静态/动态、`in_order`/`out_of_order` 下的关系，运行时融合/`runtime skip` 是否使编译期区间失效）的显式验证结论；
  3. 明确哪些现状规则（如 `oooq_memory_dependencies` 对所有队列类型无条件运行、`may_alias` 依赖 oooq）是后续设计必须兼容或主动改变的边界条件；
  4. **验证结论必须附带本次会话重新核实的 file:line 证据**，仅给出文字结论而无代码证据支撑的诊断一律视为**未完成**，不得放行 Step 3。
  未完成本步骤或未给出验证结论前，严禁进入 Step 3 的代码实现。

- **Step 3 [按领域分流设计（仅设计，不写代码）]**：下列专家只产出设计规格，代码实现统一在 Step 5 由 `@ov-gpu-refactoring-specialist`（Sonnet）按批准的设计完成。
  - **若任务涉及"锚定 processing_order 的统一生存区间计算、`alias_analysis`/`war_safe`、原位冲突解决，以及 Pattern A（并列分支）、Pattern B（Concat/Crop 融合组）、Pattern C（oneDNN `sum` 残差链）"**：调度 **`@ov-gpu-pool-liveness-specialist`（Opus）**。Pattern C 的跳数上限常量为 `program_helpers.cpp::is_direct_ancestor()` 中的 `max_ancestor_walk_depth`——**落笔前必须重新 Grep 核实其当前取值**。
  - **若任务涉及"落地 `Two-Phase-Static-Arena` 的 `cldnn::memory_manager` 双 Backend、Phase 1/2 拓扑 Tape 规划、`StorageUF` 加权并查集、MAY-alias Envelope、Plan Epoch 重规划，或动态 Shape 的 A2/B2 推导规划、编译期启发式计划与计划缓存"**：调度 **`@ov-gpu-static-arena-architect`（Opus）**。
  - **若任务涉及"解除 `is_cpu()` 过载、`PortMemoryContract` 求解器、KV Cache `APPEND_INPLACE` 专用持久域或 `ShapeBarrier`"**：调度 **`@ov-gpu-contract-state-specialist`（Sonnet）**。〔已决策（2026-11-16）：定位为辅助实现，只能在 Opus 设计已获批的前提下实现，不得独立做架构决策。〕
  - **协同边界（只实现一次）**：别名类划分、生存区间与 WAR 判定由**同一个 `alias_analysis` 引擎**产出（见下文"统一原位冲突解决"），legacy 池 restriction、Phase 0.5 校验、冲突解决 pass、生存区间、Phase 4s Arena 与 D 路线的按轮别名类都是它的消费方，任何 agent **不得**各自独立实现区间重叠/WAR 判定。
  - **oneDNN sum 判定的权威（F5，用户 2026-11 已批准审计方案）**：
    - `add_onednn_optimization_attributes`（约 `program.cpp:603`）冻结的 `fused_prims_onednn[].op_type`/`attrs_onednn` 是起点；
    - 在**单一检测点**做**唯一一次**实时重判。单一检测点是 `post_optimize_graph` 末尾：约 `program.cpp:647` 的 OOOQ BFS 与 `mark_state_init_subgraphs` 之后、F9 之后，紧接 `prepare_memory_dependencies` 之前。它不放在 `prepare_memory_dependencies` 内部，因为 dry_run 的 `#ifdef` 会跳过后者。`resolve_inplace_conflicts` 也在这里。最终 sum = 冻结为 sum ∧ 实时为 sum ∧ 外部性/WAR 检查通过；不满足即**降级为 binary_add，并重建 attrs、`fused_prims_onednn` 与 impl**（沿用 `post_optimize_weights` 的 `select_implementation`）；**只允许降级，不允许升级，不插拷贝**；
    - **调用点收口按时序分两类**：
      - **单一检测点之后**的调用点（`basic_memory_dependencies`、`pass_manager.h` 的依赖构建、`get_reused_eltwmem_idx` 的构造期与运行期调用）一律改读重冻结后的结果，不再各自实时调用 `get_add_fusing_type()`。
      - **单一检测点之前**的调用点（`reorder_inputs`、`layout_optimizer`、`prepare_buffer_fusing`，以及 `remove_redundant_reorders` 的 pre 阶段与 post 阶段 `:626/:639` 两次调用）是一致性守卫，取"冻结 ∨ 实时"作为"可能是 sum"的判定，保守方向是保留 reorder、拒绝 concat in-place。这与 F5 自身"降级为 binary_add"的保守方向相反，两者不能混用；
    - 映射键是 `mem_dep`，不是下标（`try_optimize_post_ops` 会删除 `optimized` 项；`optimized_sum` 无写入点）；
    - 模型缓存导入路径**不重算**，直接使用反序列化值（现状导入时懒初始化会覆盖 `fused_prims_onednn`，必须改掉）；binary_mul_inplace 的否决标志必须参与序列化。
  - **G7 已结案（用户 2026-11 确认）**：Pattern C 与"Arena 不拥有经 `reinterpret_buffer` 绑定的缓冲"不冲突。链首 root 不是经 reinterpret 绑定的，被 reinterpret 的是链上各 link；在 Arena 中 root 是该别名类唯一的分配者，各 link 是零分配的视图成员，该规则继续适用于 link。（注：本项目此前所说的"G7 冲突"是派生规则，并非文档 5 的 G7。文档 5 原表中 G7 指"`get_add_fusing_type()` 多处独立调用 + 运行时重绑定"，其 §12.1 已把该条改写为"binary_mul_inplace 经视图的 WAR"。）
  - **回退规则**：任何区间/Arena/原位冲突解决路径，对 `out_of_order` 队列、CPU impl（含"可能落到 CPU"的动态节点）、shape_of 子图、kv/变量和用户可见输出，一律回退 LEGACY 保守行为或返回 Unknown。视图本身不分配，跟随其别名类；只有 `alias_analysis` 给出完整的别名类时才能进 Arena，否则整个类回退。**`in_order` 下的动态节点不再一律回退**，必须走动态路径并证明收益（见"动态 Shape 路线"）。（其中只有 OOOQ 回退是文档 4 的不变式 10，其余为项目追加。）
  - **强制测试交付物**：只要改动触达 Pattern A / B / C 的判定逻辑或原位冲突处理，**必须同步在 `src/plugins/intel_gpu/tests/unit/` 下新增合成图单元测试**，具体要求见 §2。

- **Step 3.5 [专家分歧裁决]**：Step 3 中两位及以上专家（或同一专家的前后两轮）对**同一代码事实或设计决定**结论不一致时，必须先交由 `@ov-gpu-architect` 配合 `@ov-gpu-explorer` 用本次 Grep 证据裁决，并在 Step 4 材料中附"冲突表（断言/来源/代码证据/裁决）"。主会话在汇总专家输出前，须先在 `@ov-gpu-invariant-auditor` 的设计级审计中做一次交叉核对，不得把任何单一专家的代码断言当作已验证事实。

- **Step 4 [强制人工确认关卡 ⏸️]**：Step 2/3 产出的设计方案与影响范围（涉及改动的文件列表、新增抽象层对外接口、是否存在行为变更风险）必须先展示给用户，**暂停并等待用户确认**后才能继续对 `program.cpp`、`network.cpp`、`program_node.cpp`、`program_helpers.cpp`、`basic_memory_dependencies.cpp`、`skipped_branch_memory_dependencies.cpp`、`oooq_memory_dependencies.cpp`、`primitive_inst.cpp`、`prepare_buffer_fusing.cpp`、`mark_runtime_skippable_nodes.cpp`、`remove_redundant_reorders.cpp`、`add_required_reorders.cpp`、`memory_pool.cpp` 等核心编译/分配路径执行 `Edit`/`Write`，除非用户已明确说明"直接执行无需确认"。

- **Step 5 [实现与编译、单测闭环]**：Step 4 批准后，**由 `@ov-gpu-refactoring-specialist`（Sonnet）按已批准的设计实现全部源码与测试改动**，并编译、运行 `ov_gpu_unit_tests`（含新增合成图用例）与相关 `ov_gpu_func_tests`，确认零回归后才能进入下一步。若改动触及 `Two-Phase-Static-Arena` / `cldnn::memory_manager`，须运行文档 4 §7.2 测试矩阵中**与本阶段已落地组件对应的条目**，其余逐条标 N/A 并说明原因（多数组件尚不存在，全量运行不可执行）。

- **Step 6 [内存/并发安全复审]**：**必须调用 `@ov-gpu-reviewer-auditor`（Sonnet）** 针对本次改动核查内存对齐、Out-of-Order 场景下的生命周期重叠复用、多 Stream 线程安全、Host-Device 隐式同步阻塞等高危项，与 Step 7 的设计合规审计互补、不可相互替代。

- **Step 7 [强制红线审计门禁]**：**必须调用 `@ov-gpu-invariant-auditor`（Opus）**，分两种模式：
  - **设计模式**（输出设计方案后、Step 4 之前）：对照清单 A/B/C/D 逐项核查，其中清单 D 的"流程完整性"一项标 N/A；只适用于本阶段已声明覆盖范围的条目，其余标 N/A。
  - **代码模式**（Step 5/6 之后）：对照清单 A（红线）、B（10 大安全不变式）、C（接口纯洁性）、D（测试覆盖与架构前提，含流程完整性）逐项核查。

- **Step 8 [收益实测验证]**：Step 5–7 只证明正确性。每个阶段完成后，**必须调用 `@ov-gpu-memory-profiler`（Sonnet）** 按其 A/B 协议实测峰值显存、`ETC @ device` 占比、池内利用率与分配次数，并对比输出张量。
  - **基线**〔已批准〕：B0 = HEAD `f31d6d5`（Pre-0 之前），须从不含 F1–F3 未提交改动的干净工作树构建；B1 = Pre-0 合入后。各阶段对比其前一个已合入阶段。
  - **口径**：静态 `in_order` 与**动态 `in_order`** 都须证明有收益（动态路线中行为中立的步骤除外，另行注明）；OOOQ 须验证"无退化"；Pre-0 正确性修复适用"无退化"。
  - **动态路线（D 阶段）的口径**：
    - 装箱质量按轮统计，每轮报告三个数：活跃集下界 L、计划峰值、实际持有量。持有量是关键指标，因为它包含跨轮保留的部分。
    - 输出须与 LEGACY 路径逐元素一致。
    - 延迟与主机开销只记录、不作门槛（PoC 原则）。
  - 收益未兑现或输出有差异的阶段视为未完成，不得进入下一阶段。"无退化"的数值阈值待用户确定（审计建议：峰值显存 ≤ 基线 +0.5%、分配次数不增加、输出逐元素一致）。

## 已核实的架构事实（2026-11，坐标为快照，引用前须重新 Grep）

- **processing_order（下称 PO）**：
  - `calculate_BFS_processing_order()` 在 `pre_optimize_graph`（约 `program.cpp:552`）对所有队列无条件调用，post_optimize 的二次重算（约 `:647`）仅限 OOOQ；但 BFS 之后还可能被若干 pass 的 `calc_processing_order`（DFS）覆盖，所以 in_order 下最终 PO 只保证是"某个合法拓扑序"，不保证是 BFS。
  - `processing_num` 按链表位置现算（`program.hpp` 约 `:76-78`），不会过期，但每次 O(n)，区间引擎应一次性取快照。
- **`_exec_order`**：静态与动态网络都在 network 构造期一次性冻结（`build_exec_order`，约 `network.cpp:896-930`，只在构造函数调用一次），in_order 下等于主机入队序与设备执行序（OCL 与 oneDNN 共用同一个 `cl_command_queue`，运行期没有第二队列）。
  - 静态网络：等于去掉 data 与无依赖 mutable_data 后的 PO。
  - 动态网络：与 PO 只差一类节点——单用户、且用户是动态可优化 concat 的前驱，被推迟到紧挨 concat 之前。**这一移动是必要的**：第一个执行的前驱会用所有前驱的依赖 layout 一次性推导全部前驱与 concat 的 shape（`update_shape` 不递归），不移动会读到旧 layout。
  - OOOQ 下两者都不代表设备执行序。
  - loop/condition 的 body 网络有独立的 PO 与执行序，但与父网络共用 stream。
- **区间时间轴**：必须用 `_exec_order`（或 F9 之后的 PO）。按旧 PO 计算会低估被推迟前驱之输入的生存期，导致数据被提前覆盖。
- **`may_alias`**（约 `network.cpp:725-782`）的 in_order 路径依赖 oooq 的结果，因此不能在 in_order 下直接关掉 oooq。
- **现有 restriction 已是 MAY-alias envelope**：`pass_manager.h` 约 `:355-381` 对 can_be_optimized / runtime_skippable 节点按"可能跳过也可能不跳过"沿链递归。
- **运行期 skip**：被跳过的节点在 `on_execute` 中先归还自己的池块，再把输出别名到 0 号输入；skip 决策每轮可以翻转，但只影响动态节点（静态节点不进 `prepare_primitive` 的动态分支）。
- **memory_pool**：`release_memory` 在某块 `_users` 清空时直接 erase 记录，池中没有"空闲块"概念；查找是 `lower_bound` 加升序线性扫描（已是按尺寸最小可用块优先）；每个 network 一份池；0.5 阈值只对动态请求生效。
- **动态网络现状**：节点取得池块后一直持有、只增不减（只有扩容或"缩到十分之一以下且超过 `output_reclaim_min_size`"才归还重取）；但"持有"不等于"独占"：首次分配与重分配时，`get_from_non_padded_pool` 会把节点加入一个已有块的 `_users`，条件是请求 > 0.5×块、格式兼容、与块的现有用户无冲突，所以互不冲突的节点会共享块。"各节点历史最大输出之和"只是完全不共享时的上界，实际峰值介于它与区间装箱下界之间，须实测；Pattern A/B 在动态网络中同样存在；动态节点 sum 恒为 not_supported，所以 Pattern C 不存在，但每个残差多一块 binary_add 输出。动态网络不启用 record/replay。
- **动态 LLM 实测（文档 10，Qwen3-8B PA，单序列，离线回放，不是收益实测）**：
  - 运行期 shape 推导同步为 0 次。
  - 第一阶段（2×4096 输入）：现状 275 MiB，活跃集下界 180 MiB。差距来自整块复用、没有 offset（约 39 MiB）和 0.5 阈值（约 54 MiB）；restriction 只占约 0.5–1 MiB。去掉 0.5 阈值后回放峰值反而升到 345 MiB。
  - 持续生成阶段：持有 115 MiB，实际只需约 0.67 MiB，差额全部是保留。其中 96 MiB 是恰好 32 MiB 的块，被缩容规则中 `<=` 的边界条件留下（约 `primitive_inst.cpp:1365`）。
  - grow-only Arena 在该阶段持有 180 MiB，比现状更差；只有能缩容的方案才能低于现状。
  - 结论：LLM 的收益来自 offset 装箱与缩容，不来自 restriction 或区间判据。
- **CPU 判定**：`compile_graph` 在 `prepare_memory_dependencies` 之前运行并为静态节点设置 `selected_impl`，之后所有 `set_selected_impl` 调用点都在其前。编译期判定 CPU 用 `selected_impl ? selected_impl->is_cpu() : (preferred==cpu)`；动态节点运行期可能落到任一已支持的 impl，其已支持 impl 集合含 cpu 时按"可能是 CPU"保守处理。
- **Concat / Crop**：非 internal 网络中可优化节点在 `allocate_output()` 绕池（约 `primitive_inst.cpp:3241`），internal 网络走池；Concat holder 是 concat 节点本身，Crop holder 是生产者；`crop_in_place_optimization::optimize(crop_node&)` 无调用者。
- **`layout::operator==` 比较 padding**；静态 resample 的 no-op 判定因此受 padding 保护。
- **GPU 没有"检测到原位冲突就插入拷贝"的通用机制**；唯一类似的是 stateless_kv 的运行期私有拷贝（约 `primitive_inst.cpp:871-889`）。
- **规划器对比数据的出处**：MemorySolver（N=20k 约 4.4s）vs 自写 Best-Fit（约 7ms）来自一次已删除的临时脚本，单次运行、合成负载、Debug libopenvino；装箱质量相对下界分别 +4.3% / +4.6%。不足以作为选型依据。

## 统一原位冲突解决（2026-11 设计方向，经 Opus 交叉审计）

GPU 不照搬 CPU 的 `ResolveComplexInplaceConflicts`（最后统一检测、插入拷贝），因为 GPU 的原位决定散在多个 pass、Concat 的 padding 已写进前驱 layout、oneDNN 的 sum 已冻结进 attrs 并按其建好 impl。冲突分两类：

| 类别 | 例子 | 处置 |
|---|---|---|
| 角色冲突（layout 已被改写） | 视图与 Concat 槽位冲突、padding 被清掉、跨格式视图（F1/F2/F3/F7） | 只能在声明点拒绝；统一为"后执行的声明者向 AliasSink 查询角色"，不再各自打补丁 |
| 时序冲突（WAR、写入外部 buffer） | sum 写入的 buffer 仍有后续读者、sum 写入外部根或输出 | 在单一检测点（与 F5 相同，见 Step 3）由 `resolve_inplace_conflicts` 统一检测；sum 降级为 binary_add 并重建 impl（F5），binary_mul_inplace 设否决标志，可跳过节点清除可跳过标志（kv_cache 运行期无条件尝试原位，须按"总是原位"计入） |

- **`alias_analysis` 引擎**：输出别名类划分（最终标志位推导为成员权威，AliasSink 记录提供 kind/偏移/角色，两者比对构成影子校验）、PO 索引、`war_safe(W)` 三值判定（Safe / Unsafe / Unknown）与类区间。
- **`war_safe` 读者闭包**：沿视图成员递归、跳过 shape_of；写者经其他端口读到同一类也算冲突（R4）；类中有输出成员（含 mutable_data 传播的输出标记）时加一个排在最后的虚拟读者；外部根只允许已登记的持久写者；OOOQ、CPU impl、shape_of、带回边的 body 返回 Unknown。
- **Phase 0.5 原则相应调整**：角色类规则（R1/R2/V）仍只验证、修复留在声明点；时序类规则（R3/R4/R5）升级为"在单一点检测并降级"。

## 阶段化执行约定

用户任务要求"分步演进路线"，因此每次调度只推进**一个**被批准的阶段。状态标注：〔已批准〕〔建议·待批准〕。

- **Pre-0 正确性修复**（独立于内存池重构，行为变更，各自需批准；"先写单测复现再修"）：
  - **F1** reorder→permute lambda 补充 reorder/permute 输出格式检查〔已完成，2026-11-16 提交 `2bb523e677`〕。
  - **F2** 该 lambda 在 permute 的用户是已优化 Concat 时放弃优化〔同 F1〕。
  - **F3** 静态无操作 `strided_slice` 在有已优化 Concat 用户时不标跳过〔同 F1〕。
  - **F4** oneDNN sum 写入视图、而视图所有者另有读者（见 `tests/unit/passes/inplace_alias_conflict_test.cpp` 的 4 个 `DISABLED_row11_*` 用例）：等 Phase 1 的 `war_safe` 落地后实施，与 F6 共用"沿视图链回溯所有者"的 helper〔已批准方向〕。
  - **F5** sum 判定单点重冻结、只允许降级、降级时重建 impl、导入路径不重算（见 Step 3）〔已批准〕。其中结构性部分属 Pre-0：单点重冻结、单一检测点之后调用点的收口、输出标记分歧时降级、导入不覆盖。依赖 `war_safe` 的部分在 Phase 1 接入。binary_mul_inplace 否决标志的存放位置须在设计稿中写明：不得放在 `fused_primitive_desc` 上（不新增该结构的缓存字段），且必须参与序列化。
  - **F6** sum 的依赖沿视图链回溯到所有者，所有者为外部根或链上有输出时拒绝 sum〔已批准；与 F13 合并实现〕。
  - **F7** `add_required_reorders` 的 `eliminate_pad_for_onednn_impl` 可能清掉 in-place concat 前驱的 padding：先单测复现〔已批准〕。
  - **F8** resample 补 `is_output` 防护，与其他可跳过类型一致〔已批准；并覆盖静态 no-op resample 接 in-place Concat 的同类漏洞（`mark_runtime_skippable_nodes.cpp` resample 分支）〕。
  - **F9** 统一 PO：在 `post_optimize_graph` 末尾把动态 concat 前驱的移动写进 PO，使 `_exec_order` 等于过滤后的 PO；network 侧保留移动分支作为旧模型缓存的兼容路径（对已移动的 PO 幂等）；Debug 下断言两者逐项相同、无重复项；前驱有多个用户时取消该 concat 优化。并入三项先复现再修：前驱可能被执行两次、"shape 已由他人更新"标志残留、skipped_branch 在动态网络下按旧 PO 计算导致约束不足〔已批准〕。
  - **F10**（缺陷 A）Concat 前驱 P 有两个用户（concat 与 R），写者 W 以 concat 为 sum 依赖，W 写入 P 的槽位而 R 在其后读取：静默算错〔已批准立项，先单测复现〕。
  - **F11**（缺陷 C）`p_node_consumes_dep_node` 只按节点身份比较，W 经视图读到 sum 依赖同一块 buffer 时漏判：静默算错〔已批准立项，先单测复现〕。
  - **F12**（缺陷 E）reorder→permute 静态路径未检查 permute 本身是否为网络输出〔已批准立项，先单测复现〕。
  - **F13**（缺陷 F）sum 写入 input_layout（`get_add_fusing_type` 允许 users≤1 的 input_layout），之后 `set_data` 替换输入而写者不重绑〔已批准立项，先单测复现；与 F6 合并评估〕。
  - **F14**（缺陷 D）read_value 在守卫中绕过 `is_output` 与 fused 检查〔已批准立项，先单测复现〕。
  - **F16 候选**：reorder→permute lambda 把 permute 标为已优化后，`remove_redundant_reorders` 可能删掉其 reorder 搭档，留下一个没有搭档却仍被标为视图的 permute（F1–F3 收尾时在基线上观察到；F1 拦住了已知的 byxf 情形，默认格式下是否仍可达待复现）〔待复现后立项〕。
  - **F8**〔已完成，2026-11-16 提交〕。数据丢失只在 resample 由图输入/外部 buffer 直接供给时发生；concat 前驱一项为纵深防御。**F7**〔已完成，2026-11-16 提交〕：原 conv 拓扑被 `reorder_inputs` 提前插入的 reorder 挡住，改用 oneDNN gemm 拓扑复现。未覆盖：已优化（视图）reorder 作为带 padding 的输入、fsv16、动态。
  - **F10、F11**〔已完成，2026-11-16 提交，见 git log〕。后续：Step 8 统计 CNN 集（YOLO/CSP/UNet）sum post-op 数量以衡量过度拒绝；Phase 1a 用 `war_safe` 在冻结点放宽 F10；F6/F4 使用 `collect_path=true`；随 F6 补级联 concat 与 strided_slice/reorder 视图单测；`through_may_view` 由 F6/F4 使用或删除。
  - **F12、F14**〔已完成，2026-11-16 提交，见 git log〕。另记既有问题：CPU read_value impl 把 zp 拷到 `output_memory(1)` 而非 2（`impls/cpu/read_value.cpp` 约 :87），F14 后输出型压缩 read_value 更常走到该路径〔待复现〕。
  - **F17**（原 F6-mul）：`can_use_mul_inplace`（binary_mul_inplace）接受单用户 input_layout，且无所有者/输出检查；动态 realloc 重绑（`primitive_inst.cpp` 约 :1384-1398）把用户输入内存当作 FC 的 dst，LLM 式动态路径上会覆盖用户输入〔已批准立项（2026-11-16 调度官代决），排在 F6+F13 之后，先单测复现，复用 F6 helper〕。
  - **F13 注意**：修复会撤销上游 `ceedce643f`（distil-large-v2 INT8-CW 第二 token 提速 16.6%）的单用户 input_layout 例外；审计判定该例外对插件用户数值错误（插件输入总走 set_data 替换分支）。Step 8 须对该模型测延迟与对 CPU 的精度。
  - **F15 候选**：`prepare_buffer_fusing` 的 `can_optimize` 过滤对 kv_cache 也跳过 `is_output` 检查，kv_cache 为网络输出时可能同 F14 一样返回被改写的状态 buffer〔待复现后立项〕。
  - 另记录（2026-11-16 F12/F14 设计审计）：`basic_memory_dependencies` 经 mutable_data 传播的输出标记发生在 F12/F14 守卫之后，归入 Phase 0.5 R5；无用户的输出 read_value 与独立 assign 之间无序边，属时序冲突，归入 Phase 1a `war_safe`。
  - 另记录（不单独立项，作为 Phase 4s 的前置）：缺陷 B——sum 链 root 回溯遇到已优化 Concat 时走到第一个前驱。现状只是浪费，但 Concat holder 成为 Arena 别名类 root 后会变成数据损坏，**必须在 Phase 4s 之前修正**。
- **B0 基线与测试基础设施**〔已批准〕：取 B0/B1 基线；结构化分配追踪钩子；带 oneDNN（supports_immad）的测试机；真实图规模数据（供规划器选型）；动态模型逐轮活跃集测量，并入 D0（Qwen3-8B 已由文档 10 完成）。0.5 阈值的问题已由文档 10 的回放回答：去掉阈值后峰值更大，不再单列实验。
  - **B0-ARENA 大块分配可行性实测**〔2026-11-18 用户批准加入，执行者 `ov-gpu-memory-profiler`〕，结论决定 Phase 4s 的段容量 C：
    1. 在 Arc 上读出 `max_alloc_mem_size` 与 `max_global_mem_size`。
    2. 默认配置（`enable_large_allocations=false`）下分配一块略大于上限的 `usm_device`，确认 `check_allocatable` 抛异常。
    3. 在大于 4GB 的父 `usm_device` 上用 `create_subbuffer` 切出偏移超过 4GB 的子缓冲，分别跑一个 OCL kernel 和一个 oneDNN primitive；比较带与不带 `-cl-intel-greater-than-4GB-buffer-required` 时的正确性与延迟。
    - 背景（2026-11-18 核实，坐标为快照）：
      - `engine.cpp` 的 `check_allocatable`（约 :315-349）在默认配置下对单块超过 `max_alloc_mem_size` 的请求抛异常。
      - 只有 USM 分配带 `CL_MEM_ALLOW_UNRESTRICTED_SIZE_INTEL`（`ocl_memory.cpp` 约 :508），cl_mem 不带；ZE 后端无大分配处理。
      - >4GB 编译选项作用于全部 OCL kernel，有性能代价。
      - USM 的 `create_subbuffer` 只做指针加偏移，有越界检查、无对齐检查；cl_mem 子缓冲须按 `CL_DEVICE_MEM_BASE_ADDR_ALIGN` 对齐，代码不检查；image2d 不支持切分。
      - src 中没有 ZE 虚拟内存预留。
    - 默认设计假设：**分段 Arena**。每段不超过 C（C ≤ `max_alloc_mem_size`），张量不跨段，偏移对齐取 `max(sub_buffer_base_alignment, 64B)`。只有第 3 项实测证明安全且无退化时才允许 C 超过上限。

#### 统一 Arena 路线（2026-11-18 用户决定：静态网络并入同一套 Arena）

**原则**：静态网络就是"签名永远命中、只有一段"的 A2。静态网络与动态网络共用一个规划器、一套别名类和一套绑定机制，不再分阶段修补旧池使其接纳更多节点。

**已撤销的旧阶段**：

| 旧阶段 | 去向 |
|---|---|
| 1b `may_alias` 消费区间 | 并入 Phase 4s：`may_alias` 对 Arena 张量改查计划中的 offset 是否重叠 |
| 1c-static 区间取代 oooq | 并入 Phase 4s：Arena 张量不再使用 oooq restriction；OOOQ 网络整网走 LEGACY |
| 1c-dyn 运行期出借/归还 | 由 D 路线取代 |
| Phase 2 Concat holder 入池 | 并入 Phase 4s：holder 就是别名类的 root |
| 3b sum 链首 root 进池 | 并入 Phase 4s：root 就是别名类的 root，类区间取全部成员区间之并，"整链 restriction 并集"由别名类划分保证 |

**新顺序**：

Pre-0 → B0/D0 → Phase 0.5 → Phase 1a → Phase 0 → **Phase 4s** → D3 → D4 → D5 → D6 → N2 → 〔可选〕Phase 3a

其中 D1、D2 行为中立，可以与 Phase 1a、Phase 0 并行推进；N1 须在 Phase 4s 设计定稿前完成。

**各阶段**：
- **Pre-0**：内容与顺序不变，见上文。
- **B0/D0**：B0 基线、B0-ARENA 和 D0 统计合并为一个阶段（见上文与"动态 Shape 路线"）。另增两项统计：
  - 多输出节点的个数，以及 idx>0 的输出字节数；
  - 网络输出的字节数占峰值的比例。
- **Phase 0.5 `alias_analysis` + 影子校验**〔已批准〕：
  - 须先由 Opus 产出独立设计稿，内容包括：R1 完整矩阵、R4、`war_safe` 签名与三值语义、OPAQUE 别名清单、R5 补 `is_output`、V 规则命中后的处置。设计稿通过后才能进入 Step 4。
  - 只验证、不改行为；Debug 默认开，Release 需显式开关；oracle 在第一次 `execute()` 结束时采样。
  - 引擎须同时支持两种输入：编译期决策（静态网络与 LEGACY 用的 MAY envelope）和本轮决策（D 路线）。
- **Phase 1a** `war_safe` + `resolve_inplace_conflicts`（in_order 静态）〔已批准〕：接入 F4、F5 的 WAR 部分、F6、F8，新增 binary_mul_inplace 否决标志；前置 F9。它决定原位写是否合法，Arena 直接消费它的结论，所以保留。
- **Phase 0 `AliasSink`**〔已批准〕：覆盖全部声明点（prepare_buffer_fusing、handle_reshape、remove_redundant_reorders、select_preferred_formats、mark_runtime_skippable、构造期视图），行为中立，附分配序列二进制对比测试（需 B0 的追踪钩子）。它是 Phase 4s 别名类的影子校验。
- **Phase 4s 静态 Arena**〔已批准方向；设计稿待 Step 4〕：
  - 适用范围：in_order 静态网络。network 构建时由 `alias_analysis` 给出别名类，在 `_exec_order` 上取类区间；内部缓冲作为独立张量参与装箱，区间为 `[n, n]`；离线装箱，求解器按 B0 的真实图数据选定；按 B0-ARENA 的结论分段，计划不序列化（模型缓存导入后重算）。
  - 一次覆盖 Pattern A（区间代替两两冲突）、Pattern B（Concat/Crop holder 作为 root）和 Pattern C（sum 链 root 作为 root，不再 `can_share_buffer(false)` 直配）。
  - 前置：缺陷 B 已修、F5 全部完成、F7、Phase 1a、Phase 0.5 与 Phase 0 影子在目标模型上干净、N1 结论已出。
  - 审计要求：对文档 4 的不变式 3、4、5、9 给出具体机制，否则退回。
  - Go/No-Go：
    - SeedVR2 DiT whole 的 ETC 大幅下降（3b 估算约 14GB，以实测为准），峰值同步下降，输出逐元素一致；
    - CNN 回归集（VAE、UNet、ResNet、YOLO）峰值不升，kernel 数不变；
    - OOOQ 与动态网络比特级无变化。
- **D3–D6**：见"动态 Shape 路线"。D4 复用 Phase 4s 的规划器与绑定机制，动态网络只多出"每轮推导、签名查缓存、容量策略"三部分。
- **N1 多输出节点**〔新增；先核实〕：
  - 由 `@ov-gpu-explorer` 核实三件事：现状下 idx>0 的输出由谁分配、是否进池、restriction 是否覆盖；Opus 复核。
  - 设计要求：每个输出端口都是独立张量，各自有区间、各自进入装箱；每个端口单独判断是否为网络输出、是否被视图别名。
  - 核实之前，多输出节点 idx>0 的输出一律走 LEGACY。
  - 须补合成图单测：两个输出寿命不同；一个输出是网络输出、另一个不是；dynamic_quantize 的 scale/zp 输出（文档 10 中跳过时只释放 output[0]）。
- **N2 输出域**〔新增；建议·待设计，排在 D6 之后〕：
  - 问题：用户可见输出的生命周期跨越本轮，现在一律走 LEGACY，不参与复用。
  - 设计方向（参考文档 7 的 ExternalBindings）：单独设一个输出域，与 Activation Arena 分开。
    - 用户通过 `set_output` 提供了 buffer 时，产出节点直接写入，不额外分配。
    - 没有提供时，输出域按轮分配；上一轮的输出在用户取走前不得被覆盖。
    - 输出节点在本轮内被其他节点读取时，其区间延长到本轮末尾，但不占用 Activation Arena。
  - 输出被视图别名时（例如 reshape 之后才是输出），整个别名类归入输出域。
  - 是否值得做，看 D0 统计出的输出字节占比（例如 LLM 的 logits、视频模型的帧输出）。
- **〔可选〕Phase 3a 去掉 sum 跳数上限**〔原已批准；降为可选，排在 Phase 4s 之后〕：
  - Phase 4s 落地后，sum 链断开只会多出一个 binary_add 输出张量，这个张量在 Arena 中照常复用，不再造成池外直配。所以 3a 剩下的收益只是这一个张量和一次原位写。
  - 风险 R1（CNN 连带损失）不变。
  - 只有当 Phase 4s 的 Step 8 显示剩余差额值得时才启动，仍采用 M1 + M4。

每阶段独立走 Step 1–8，独立可测、可回退；上一阶段 Step 8 未通过不得开始下一阶段。

### 动态 Shape 路线：A2/B2 推导规划 + 编译期启发式计划〔用户 2026-11-18 定方向；D 阶段各自待 Step 4 批准〕

本节取代旧的 S0–S6 与 E 路线。旧路线中下列内容与新方向冲突，**已删除，不得再引用**：
- 运行期按最后消费者出借/归还（旧 S3 / 1c-dyn）；
- 只增不减的 Arena 与几何分档（旧 S4；文档 10 已证明 grow-only 在 LLM 持续生成阶段比现状更差）；
- "启发式 Arena + 第二块"；
- 以首 token 延迟作门槛（旧门 G2）。

**谱系**（文档 8 §8.3）：

| 尺寸可知时机 | 同步点 = 0 | 同步点 > 0（检测、分割、MoE、控制流） |
|---|---|---|
| 逐节点（A1） | HWM 计划 + 中途溢出 | 同左，另加 DEFERRED（B1） |
| **先推导（A2）← 本项目** | 整图推导 → 重放或重规划 → 入队 | **按段推导 → 在线续排 → 入队（B2）** |

A2 是 B2 只有一段时的特例，因此只实现 B2 一套机制。

**每轮流程（in_order）**：
1. **推导**：按 `_exec_order` 对当前段做整图推导，得到每个张量的精确 layout，同时确定 impl（`update_impl` 前移，按轮冻结）、内部缓冲尺寸、runtime skip / in-place 决策和期望的权重 layout。推导阶段不入队任何 kernel。
2. **取计划**：用"轮签名"查计划缓存。轮签名包括全部张量的字节尺寸、别名决策、impl 选择和内部缓冲尺寸。命中则重放 offset，未命中则对本段做装箱规划并写回缓存。
3. **绑定并入队**：按 offset 在 Arena 上切子缓冲，绑定到各节点，然后入队本段。
4. **到达切点**：已有的 `finish()` 加数据回读后，进入下一段。跨段存活的张量保留 offset；新段的张量只在当前空闲区间里在线续排；空间不够时追加 chunk。本轮结束后，下一轮按观测到的尺寸合并为单个计划。

**上界问题如何消解**：
- 不依赖上界。每段入队前尺寸已精确可知，所以 Arena 容量在入队前就能定好，不存在执行中溢出。
- 设备显存确实不够时照常报 OOM，这不是规划器的问题。
- 文档 9 已说明：LLM IR 的序列维通常无界，GenAI GPU 路径不提供上界，所以基于上界的方案（文档 7 的 envelope）不能作为主路线。

**编译期启发式计划**：
- 用途：`compile_model` 时用一个初始的启发式 shape 做一次推导与规划，**只产出计划并缓存，不做物理分配**（与文档 7"编译期零物理分配"一致）。首轮命中时，可直接按该计划一次性分配 Arena。
- 来源优先级：用户配置的 shape profile（参考文档 7 的 `arena_shape_profiles`）> IR 中有界维度的取值 > 都没有则不生成。具体取值规则在 D6 设计稿中确定。
- 硬约束一：启发式推导必须是纯计算，不得改写节点 layout、impl、kernel 与 padding。所以 D6 依赖 D1 的纯函数推导。
- 硬约束二：启发式计划不是上界，不得用它限制或撑大运行期 Arena。不命中时直接走 A2/B2。
- 已知局限：编译期动态节点只有 shape-agnostic impl，运行期换成静态 impl 后内部缓冲尺寸会变，签名随之不同。所以这个计划的命中率须由 D0 实测，不能预设。

**命中与 Arena 容量策略**：
- 命中判定默认采用签名完全一致。可选放宽为"本轮每个张量都不大于原槽位，且浪费比例不超过阈值"；放宽只能在开关后启用，且须按轮报告浪费量。
- 容量规则：计划峰值大于当前容量时换一块更大的 Arena；计划峰值小于当前容量的 r 倍时缩容，r 可配置。这一规则与 LLM 专有的阶段概念无关，对非 LLM 同样适用。
- 旧 Arena 必须等上一轮执行完才能释放。
- Arena 按 B0-ARENA 的结论分段，单段容量不超过 C，张量不跨段。

**不进 Arena、走 LEGACY 的部分**：沿用 Step 3 的回退规则（OOOQ、CPU impl 及可能落到 CPU 的节点、shape_of 子图、kv/变量、用户可见输出、外部输入），另外加上：
- loop/condition 的 body 网络：外层把控制流节点当作切点；
- 推导失败或签名中出现 Unknown 的节点。

视图是零分配成员，跟随别名类 root 的 offset。
- **别名类按轮确定**：推导阶段已经做出本轮的 skip、in-place concat/crop 与 kv_cache 原位决策，所以动态网络的别名类按本轮实际决策划分，不再使用编译期"可能跳过也可能不跳过"的 MAY envelope。MAY envelope 只留给 LEGACY 路径。
- 按轮划分与编译期划分由同一个 `alias_analysis` 引擎产出：引擎以本轮决策为输入，不另写一套（见"协同边界"）。
- 计划不序列化。

**对文档 8 第 8 章的修正（坐标为快照）**：
- 切点除 `has_runtime_deps` 外，还有 `do_runtime_skip_gather` 的 `finish()` 加索引回读（约 `primitive_inst.cpp:2016-2018`）。
- `update_weights` 在 prepare 阶段会入队 reorder kernel（约 `:3139`），所以推导阶段只能确定期望的权重 layout，reorder 的入队留在执行阶段。
- `update_impl` 必须进推导阶段，因为内部缓冲 desc 与 FC fake alignment 都依赖 impl。impl 按轮冻结，异步编译完成后的切换推迟到下一轮。
- Tape Best-Fit 是在线算法，所以同样的尺寸和顺序下，A2 与"在 Arena 偏移空间上逐节点在线装箱"得到的 offset 相同。A2 独有的是三点：入队前就定好容量，所以不溢出；缩容没有滞后；可以换用非在线求解器（例如按尺寸降序的离线装箱）。

**推导范围的事实（LLM 持续生成阶段，已复核，坐标为快照）**：
- 输入 layout 不变的节点在 `update_shape` 中比较完 layout 就提前返回（约 `:438-466`）。
- past_len 只影响 kv_cache、kv 到 SDPA K/V 输入之间的节点、mask 路径和 shape_of 子图。文档 10 中稳态每步只有 36 个 PA 节点重算 layout。
- 例外一：异步编译过渡期，shape-agnostic impl 每步都进 `update_impl`。
- 例外二：未覆写 `get_shape_infer_dependencies()` 的类型（如 `slice`），输入来自 GPU 时每步都会重算并 `finish()`。
- record/replay 对动态网络整网关闭（约 `network.cpp:1215/:1218`）。

**D 路线**：每阶段独立走 Step 1–8。D0–D3 行为中立。

| 阶段 | 内容 | 门槛 |
|---|---|---|
| D0 | 基线与插桩（并入 B0）：每轮的切点数（区分 `update_shape_sync` 与 gather 回读）、SHAPE_CHANGED 节点数与类型分布、异步编译过渡期的节点数与持续轮数、`slice` 类每轮重算、skip/in-place 翻转次数、活跃集下界 L、现状持有量；主机耗时只记录。Qwen3-8B 已有文档 10，须补齐非 LLM 与其余 LLM 形态 | 数据齐全 |
| D1 | 纯函数推导 `infer_layouts()`：从 `update_shape` 中拆出来，连同 skip/in-place/reorder/permute 的决策谓词。整图影子推导，与实际逐节点结果逐项比对 | 目标模型集上零差异 |
| D2 | 影子规划器：用 D1 的结果与 `alias_analysis` 在 `_exec_order` 上的区间做每轮装箱；独立校验器检查"offset 重叠的张量区间不相交"；按轮报告计划峰值、L 与现状持有量 | 校验零违例；给出装箱质量数据 |
| D3 | 决定/应用拆分：`update_impl` 前移并按轮冻结，`update_weights` 拆成推导与执行两部分，concat/crop/reorder/permute 的决定与应用分开 | 输出与执行序列逐项不变 |
| D4 | A2 单段落地：只对切点为 0 的网络启用，开关默认关闭；包含计划缓存与容量策略 | 输出逐元素一致；持有量低于现状 |
| D5 | B2 分段：在切点处分段，跨段在线续排，追加 chunk，迭代边界合并 | 同 D4，覆盖有切点的模型 |
| D6 | 编译期启发式计划与缓存 | 命中率数据；未命中时与 D5 结果一致 |

**D 路线的前置**：
- D2 依赖 Phase 0.5 的 `alias_analysis` 与 F9 统一后的 PO。
- D4 依赖 Phase 4s 已落地（复用其规划器与绑定）、Phase 0.5 影子报告在目标模型上干净、F17 已修。
- 动态节点的 sum 恒为 not_supported，所以 Phase 1a 的 `war_safe` 不是 D4 的前置。但 binary_mul_inplace 的动态重绑定必须按"前向约束"处理。

**目标模型**（D0 确定清单，同时用于判断切点数与签名命中率）：
- 稠密 LLM（SDPA 与 PA 两条路径）、MoE、VLM；
- 非 LLM：FlashVSR、SeedVR2（分辨率与帧数动态）、minimaxh3。
- 需用户提供：每个模型的输入 shape 分布、部署后端、多实例配置。

### Pattern C / Phase 3a 的风险（2026-11-16 Opus 分析，坐标为快照）

> 2026-11-18 起，原 3b（root 进池）并入 Phase 4s，3a 降为 Phase 4s 之后的可选项。下文中的"3b"一律指 Phase 4s 对 sum 链别名类的处理。R1–R5 仍是 3a 的风险；R2 中关于 3b 的部分是 Phase 4s 的硬前置。

**收益从哪来**：
- SeedVR2 DiT 中约 15.1GB 的 ETC，不是 binary_add 输出本身落到池外。机制是：attn 残差退化为 binary_add 后，它的输出成为下一个 MLP 残差 sum 的依赖；`basic_memory_dependencies.cpp`（约 `:77-81`）把它认定为新链的 root 并设 `can_share_buffer(false)`，于是在 `allocate_output()`（约 `primitive_inst.cpp:3241`）直配。
- 两条路任一都能回收大部分：
  - 3a 去掉跳数上限：全网合成一条链，约省 14.6GB；
  - root 作为 Arena 别名类 root（Phase 4s，原 3b）：相邻 root 交替复用，约省 14GB，类区间须是整链成员区间之并。
- 由于 Phase 4s 已能回收约 14GB，3a 剩下的增量只有约 0.6GB 量级（估算，以实测为准）。
- 7.6GB 的窗口 Concat FP16 由 Phase 4s 的 Concat holder 处理，不计入 3a。

**第一道门槛（2026-11-16 已回答）**：离线解析 `/home/ceciliapeng/openvino/seedvr2_dit_only_480p_case_20261010.whole/gpu_exec_graph_dump.dit/runtime_gpu_graph_stream0_0_gpu_exec_graph.xml`（2026-10-10 导出，按 input port 0 回溯，脚本 `/tmp/dit_walk/walk.py`）：
- **attn 残差**从 FC `MatMul_2543` 到残差源 `MatMul_1682` 共 **25 跳**，路径依次是：Gather → Concat(窗口合并) → Crop → Reshape → Permute → Reshape → SDPA → Concat → Crop → Concat(45 users) → RoPE → Reshape → Permute → Reshape → Crop → Permute → Reorder → Reshape → RMS → Gather → Reshape → Crop → QKV FC → RMS → 残差源。
- **MLP 残差** 3 跳。

结论有两点：
- 当前上限 7（最多 6 跳）下 attn 残差**仍然断链**，10-10 之后的 pass 改动不可能把 25 跳缩到 6 跳以内；
- 整条路径沿 dep(0) 可达，3a 去掉上限后两条链能接上。

上游注释（`program_helpers.cpp` 约 `:112-138`）所说"attention 残差位于第 6 跳"，指的是标准 MHA，不适用于 DiT 的窗口注意力加 RoPE 结构。合成图单测（设计中）定位为防回归与刻画现状。

| 风险 | 内容 | 严重性 |
|---|---|---|
| R1 早期决策的连带损失（核心） | 三个调用方在 F5 单一检测点之前就按 sum 做了不可回收的决定。(a)(b) 只在冻结点（约 `program.cpp:603`）之前运行；(d) 在冻结点之前（`:587`）和之后（`:626/:639`）都会运行。另一个早期调用方 (c) `reorder_inputs.cpp` 约 `:695` 只对 binary_per_tensor 插 reorder，sum 不受影响：(a) `prepare_buffer_fusing.cpp` 约 `:213-216`，前驱带 sum 时拒绝 concat in-place，concat 变成真实拷贝；(b) `layout_optimizer.cpp` 约 `:214-220`，sum 且格式不同时不融合输入 reorder；(d) `remove_redundant_reorders.cpp` 约 `:307-317`，dep 带 sum 时保留 simple reorder（条件比注释宽，不限输出 reorder）。去掉上限会让更多残差判为 sum，这些损失在 F5 降级时**无法回收**：concat 已物化、padding 已传播、reorder 已保留并编译出 impl，F5 只能重建本节点的 attrs 与 impl。这三处同时是一致性守卫（防 SUM_DST+SLOT、保证 sum 所需的同格式、防止 sum 写者变成输出），不能简单放宽 | DiT/LLM 风格 FC 块基本无损（残差不进 concat、格式统一）；**静态 CNN 与混合结构高危**：UNet/YOLO/CSP 的"conv+残差→concat"、fsv16↔bfyx 边界、SD UNet 的 transformer 块、SeedVR2 VAE |
| R2 正确性（漏标方向） | 去掉上限本身不新增"直接用户"的 WAR：target 必须恰有 2 个用户，`p_node_consumes_dep_node` 排除了 dep(0) 就是 target 的情况，所以另一个用户必在 p 的 dep(0) 回溯链上、先于 p 执行，与深度无关（注意区分"dep(0) 回溯链"与"残差复用链"）。target 本身是否为网络输出没有检查（只查了 p），这是既有缺口，归 F6。风险集中在：另一个用户是视图时视图的其他读者（F11）；链上中间节点的其他用户（现状只给每一跳的直接用户加 restriction）；Concat holder（缺陷 B）与第二读者（F10）。Phase 4s 的硬前置（原 3b）：root 的类区间必须覆盖链上全部成员及其全部用户 | 静默损坏 |
| R2' Unknown 的处理 | OOOQ 下保留带上限 7 的旧谓词，两个谓词按队列类型并存；队列类型在 pre_optimize 时已可读，所以早期调用方也必须用同一个按队列选择的谓词。CPU/shape_of 读者那一跳回退到旧的有界结果 | — |
| R3 判定一致性 | 早期调用方、冻结点、F5 必须共用同一个结构谓词 S（去掉上限、不含 WAR），F5 取"冻结 ∧ 实时 S ∧ WAR"，冻结之后不会升级。但早期判定到冻结之间仍有改图（`add_required_reorders` 约 `:595` 插 reorder 会改变 dep 的用户数与格式，`fuse_primitives_with_layout` 约 `:599`），可能出现"早期判非 sum 并放行了 concat in-place / reorder 消除，冻结时却判成 sum"的升级。多数情况由 sum 条件中"padding 与 format 相同"自保；兜底是 Phase 0.5 的 R1 捕获 SUM_DST+SLOT，以及在 program 级侧表记录"早期已按非 sum 放行"，冻结时取 S ∧ ¬已放行（侧表不是 `fused_primitive_desc` 字段，实施前需单独批准） | 静默损坏（若漏守） |
| R4 编译时间 | F5 在 `compile_graph` 之后，每次降级都要重建 oneDNN primitive；去掉上限后 `is_direct_ancestor` 找不到 target 时会走到输入端，O(N)，被多个 pass 反复调用，最坏 O(N²)。缓解：以 PO(target) 截止并预建序号表，或对结果做缓存 | 性能 |
| R5 测试与数值 | `add_onednn_optimization_attributes_test.cpp` 的 `ancestor_walk_depth_boundary` 钉死了上限，需改写为"in_order 深度 8/16 判 sum、OOOQ 保持 7/8 边界"；该文件 `:67/:108/:148/:241/:254` 的静默 return 要改为 `GTEST_SKIP`；`tests/unit/fusions/*` 的 fused 计数可能变化。sum 与 binary_add 理论上数值一致，但 post-op 顺序与 sum 只要求"字节宽相同"可能带来差异，须做输出比对 | 回归 |

**缓解方案**：

| 方案 | 结论 |
|---|---|
| M1 乐观早判 + F5 晚降级 | 安全（只降），但 R1 的损失不可回收 |
| M2 早期保守 + 晚升级 | **违反 F5"只降不升"**，且要证明升级目标不是槽位、不是输出、格式一致，不采用 |
| M3 早期调用方改问"可能成为 sum 写者"并按 WAR 分级 | 依赖分级准确度，复杂度中高 |
| M4 配置开关 / 按拓扑灰度 | 只在开启的模型上承担 R1 |
| M5 冻结后、compile 前先做一次 WAR 预降级，F5 兜底，侧表单调守护，OOOQ 保持上限 7 | 安全，减少 impl 重建。但**与已批准 F5 的"唯一一次实时重判"冲突**；预降级发生在 F9 与 post 阶段视图生成之前，此时 PO 也不是最终值 |

**Opus 的推荐是 M1 + M4 + M5**：新谓词只在 `in_order`、静态网络、开关开启时生效，默认关闭；Phase 3 拆为 3a（去掉上限）与 3b（root 进池；2026-11-18 起并入 Phase 4s）。**M5 与"program 级侧表"都需要用户单独批准**（见文末"待用户决策"）：M5 要修订 F5 的唯一性条款，侧表本质上是新的判定缓存。未获批准时只采用 M1 + M4。**（2026-11-16 决策：M5 与侧表暂不批准，采用 M1 + M4。）**

**Go / No-Go（Step 8）**：
- SeedVR2 DiT whole：ETC 下降 ≥ 14GB，峰值同步下降，输出无差异。
- CNN 回归集（VAE、UNet、ResNet、YOLO）：concat in-place 数、存活 reorder 数、kernel 总数都不增加；若增加，延迟退化须 ≤ 1%，否则对该类模型关闭。
- 编译时间增量 ≤ 5%，F5 降级数接近 0。
- OOOQ 与动态路径：比特级无变化。

**前置**：现版本 DiT 重新 dump（第一道门槛）；Phase 0.5 影子报告干净；Phase 1 的 `war_safe` 落地；F5、F6、F10、F11 已实现并有测试；Phase 4s 已落地且 Step 8 显示剩余差额值得做；基线已保存。G7 已结案，不再是前置条件。

**新增单测**：深度 8/16/32 的残差链（in_order 判 sum、OOOQ 判 binary）；另一个用户是视图且视图的其他读者晚于 p → 降级；早期判非 sum、`add_required_reorders` 之后变为 sum → 不得升级；前驱带 sum 且是 concat 前驱 → 统计连带损失；输出 reorder 不被 post 阶段删除；（归 Phase 4s）sum 链别名类区间覆盖链上每个成员的用户；动态节点恒为 not_supported；CPU/shape_of 读者回退到有界结果；binary_mul_inplace 既有风险；F5 降级后 attrs、`fused_prims_onednn` 与 impl 三者一致。

## 2. 单元测试强制要求

任何触达 Pattern A/B/C 判定逻辑或原位冲突处理的改动，必须在 `src/plugins/intel_gpu/tests/unit/` 下补齐以下合成图单元测试，不得以"现有测试已覆盖"为由省略。**各条按本阶段已落地的组件适用**，例如 F5 或 3a 不要求测 Phase 4s 才有的 root 进 Arena；不适用的条目标 N/A 并说明原因。依赖 oneDNN 的用例必须用 `GTEST_SKIP`（不得静默 return），在无 `supports_immad` 设备上 SKIP 时计为"未覆盖"，不得计为通过；因此带 oneDNN 的测试机是 Pattern C 相关阶段放行的前置条件（列在 B0 中）。

- **Pattern A（并列分支 / `oooq_memory_dependencies`）**：
  - 构造 N 个共享同一上游输入、彼此结构不可达的兄弟分支。验收是**可靠性**而非"与旧闭包完全一致"（区间法本来就比闭包宽松）：区间法判为不冲突的每一对，在 in_order 静态执行序下真实生存期不交（可用模拟执行作 oracle）；与旧 `user_bitmap` 闭包的差集逐项列出。
  - 专项覆盖"多输入汇聚"边界（Rule 2 Caveat）：A→X→B、A→u、B→u 结构下 (A,B) 现状疑似未加冲突，须验证不会发生别名穿透或生命周期漏判。
- **Pattern B（Slice→Concat / Crop 原位融合）**：
  - 区分 Concat（holder 是 concat 节点自身，非 internal 网络绕池）与 Crop（holder 是生产者）。
  - 静态 Shape 下验证融合组所有前驱的 restriction 与生存区间被正确并集到 holder，且 Concat holder 通过池或 `memory_manager::acquire()` 分配而非直接 `_engine.allocate_memory()`。
  - 动态 Shape 下验证 `do_runtime_in_place_concat()` 按"融合总会发生"的最坏情况静态合并区间，不发生运行时越界复用。
- **Pattern C（oneDNN `add_fusing_type::sum` 残差链）**：
  - 构造深度超过当前 `max_ancestor_walk_depth`（落笔前须 Grep；截至 2026-11 为 7）的残差链，验证基于 PO 的 WAR 判定能正确识别可复用场景。
  - 冻结值与实时值分歧：危险方向是"冻结为 sum、实时判为非 sum"（attrs 里有 sum 而 dst 未别名到被累加张量，静默算错）。须构造：mutable_data 输出在 sum 标记后传播输出标记；post 阶段生成视图；`post_input_reorder` 使跳数 +1；模型缓存导出再导入。断言 F5 之后 impl post-op、`fused_prims_onednn`、`share_buffer` 三者一致，分歧只会导致降级。可用白盒钩子在冻结后篡改图。
  - binary_mul_inplace（FC）经 `get_reused_eltwmem_idx` 返回索引而 `basic_memory_dependencies` 未标记的既有风险。
  - 动态节点恒 `not_supported` 的既有边界须被保留（守护用例，不需要 oneDNN 设备）。
  - 完整残差复用链 `{root -> S_1 -> ... -> S_k}`：链首 root 恢复 `can_share_buffer(true)` 前必须并集整链全部成员的 restriction 与全部 user（现状只限制每个 link 的直接 user），并携带合并区间 `[birth(root), max_death(S_k)]`。
- **原位冲突（F4/F6/F10–F13）**：sum 写入 reshape/crop 视图且所有者另有后续读者；Concat 前驱有第二读者且 sum 写入 concat；sum 依赖为 input_layout 或其视图并 `set_data` 换输入；视图链上有网络输出；写者经另一端口读本类；read_value 为网络输出；reorder→permute 中 permute 为网络输出。每项断言输出与参考一致，修复后断言降级或拒绝发生。

所有合成图测试必须覆盖刚性排除边界（见 Step 3 回退规则）：节点对中至少一方为 CPU impl 或 `is_in_shape_of_subgraph()` 时，必须验证**仍保留原有保守隔离行为**。编译期判定 CPU 用 `selected_impl ? selected_impl->is_cpu() : (preferred==cpu)`（`compile_graph` 已在 `prepare_memory_dependencies` 之前为静态节点选好 impl），运行期用 `_impl->is_cpu()`。

## 待用户决策 / 待核实

- **已决策（2026-11-16，用户授权调度官按最佳选择代为决定）**：
  - contract-state-specialist：**辅助实现**。只在 Opus 设计获批后按规格实现其领域的代码，不做架构决策；其领域的设计由 `ov-gpu-static-arena-architect` 或 `ov-gpu-pool-liveness-specialist` 产出。
  - Step 8 "无退化"阈值：峰值显存 ≤ 基线 +0.5%；分配次数不增加；输出逐元素一致（浮点路径变化处允许与参考实现同等的容差，须注明）；延迟退化 ≤ 1%。
  - B0 批准：B0 = `f31d6d5` 干净工作树，B1 = Pre-0 合入后；当前测试机（Intel Arc，supports_immad=true）即 oneDNN 测试机。顺序已由"统一 Arena 路线"取代（2026-11-18）。
  - （2026-11-18 更新）Phase 1 只保留 1a；1b、1c-static、Phase 2、3b 并入 Phase 4s，1c-dyn 并入 D 路线。
  - 动态 Shape 路线（2026-11-18 用户定方向，取代旧的 S/E 路线）：A2/B2 推导规划加编译期启发式计划，按 D0 → D6 推进，详见"动态 Shape 路线"。
  - 动态 PoC 不以延迟为门槛：上面 Step 8 阈值中的"延迟退化 ≤ 1%"只适用于静态与 OOOQ 的无退化验证。
  - F6、F7、F8 立项（F6 与 F13 合并实现，F6 的视图链 helper 供 F4 复用）。
  - 缺陷 D 立项为 **F14**（read_value 在守卫中绕过 is_output 与 fused 检查），先单测复现再修。
  - Phase 3a 采用 M1 + M4（开关默认关闭，仅 in_order 静态生效）；2026-11-18 起降为 Phase 4s 之后的可选项。
  - **M5 与 program 级侧表暂不批准**：保持 F5"唯一一次实时重判"不变。3a 落地后若 Step 8 实测 F5 降级数或 impl 重建导致编译时间超标（> 5%），再单独提出。
  - 规划器选型推迟到 Phase 4s，以 B0 的真实图数据决定；静态网络与动态网络共用同一个规划器。
  - `kernel-graph-opt` 去掉 Edit 权限，保持"唯一实现者"规则。
- **Pre-0 执行顺序**：F1–F3 收尾并提交（已完成 `2bb523e677`）→ F12、F14（小、互相独立）→ F10、F11 → F6+F13 → F7、F8 → F5 结构性部分 → F9。每项各自走"先单测复现 → 修复 → Step 6/7 → 提交"。F4 与依赖 `war_safe` 的部分仍等 Phase 1a。
- **前向约束（2026-11-16 F17 审计）**：动态 realloc 只在 `IMPL_CHANGED` 或显式请求时重绑 `get_reused_eltwmem_idx` 用户的 dst；任何在不改变 FC shape 的情况下改变依赖内存的后续改动（Phase 4s 与 D4/D5 的 Arena 绑定）都必须同时重绑这些用户。
- **F17.3 与 F5**：F17.3（静态节点读取冻结的 `fused_prims_onednn` 判定 binary_mul_inplace 是否别名）是 F5"检测点之后调用点读重冻结结果"的 mul 部分；F5 必须保持该门控读取重冻结列表。binary_mul_inplace 否决标志仍归 F5/Phase 1a。
- 待核实（只读审计未能确认，实施前须复现）：各 F 项的实际后果；缓存导入后三份 sum 判定错配的表现；节点与自身 internal buffer 的冲突（自身 uid 不进 restriction）；Rule 2 caveat 是否真的漏判；恒等 reorder（含 padding）能否选到 impl；清除动态 Concat/Crop 可优化标志后前驱动态 padding 是否正确；LoRA 在 adapter 非空时是否原位写入；USM 释放是否阻塞；dynamic_quantize 的可跳过标志由谁翻转；`basic_memory_dependencies` 用 `eltw_dep != 0` 判"已找到第一个 sum"在 outer_dep 下标为 0 时失效（潜在）。
