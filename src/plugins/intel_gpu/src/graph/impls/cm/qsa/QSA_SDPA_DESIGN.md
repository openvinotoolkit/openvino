## 15. 最终设计基线：分层 QSASDPA + IndexerCache

  本章是当前讨论版本的**最终设计基线**。它把前文的探索、取舍和修正收敛成一个
  可以直接用于人工评审、接口设计和后续实现拆分的方案。若前文历史章节与本章有
  冲突，以本章为准。

  ### 15.1 设计目标

  最终方案满足以下要求：

  1. 每个 transformer layer 拥有独立的 `IndexerCache`，不同层之间不共享 indexer
     cache memory 或 state。
  2. `QSAIndexerCache` 在图级声明 raw-K/summary 的 Variable 和 state identity；其
     runtime state 由执行 `cldnn::qsa` 的 `qsa_inst` 持有，负责 cache memory 的
     申请、动态扩容、搬迁、shape/capacity/valid-length 管理、reset、trim、beam
     metadata 和 release。该分层与 `KVCache`/`kv_cache_inst` 一致。
  3. `IndexerCache` **不计算新的 indexer cache 内容**，不执行 projection、RMSNorm、
     RoPE、raw-K pooling 或 summary 生成。
  4. `QSASDPA`/`cldnn::qsa` 负责当前调用的新数据计算和 cache 更新：indexer Q、
     raw-K、完成 block 的 summary，以及 Q2/Q3 attention 读取；cache memory 和
     logical state 仍由同一个 `qsa_inst` 内的 runtime state 管理。
  5. `reserve` 在 QSASDPA 执行前根据运行时 token/block 信息一次性计算新 cache
     长度、capacity 和写入范围；不需要独立 `commit` 节点，也不需要 runtime
     finalization。
  6. 当前 paged `QSA` 路径保持不变；动态 `QSASDPA` 使用连续 dense cache 和直接
     stride 寻址。
  7. `IndexerCache` 的 reserve/扩容/搬迁是同步 CPU 管理操作；只要 reserve 返回，
     QSASDPA kernel 所需的 descriptor 和写入范围就已经准备完成，不为 CPU manager
     额外建立 `IndexerCache`↔`QSASDPA` event。GPU 侧仍保留 QSASDPA completion
     event，用于后续 GPU 使用、reset/trim/release 和输出依赖，不增加 cache 内容
     检查阶段。

  ### 15.2 每层的组件关系

  ```text
               layer l

  main K/V projection -----------------> existing KVCache[l]
                         |
                         | dynamic dense main K/V
                         v
  hidden[l] -------------------------> QSASDPA[l]
                         ^
                         |
                  writable descriptor (reserve 同步返回)
                         |
                    IndexerCache[l]
                  raw-K state + summary state

  QSASDPA[l] outputs:
    attention output[l]
    hidden/output for layer l + 1
    qsa_event[l]

  qsa_event[l] -> hidden_ready[l + 1] -> QSASDPA[l + 1]
  ```

  主 K/V 始终由已有 `ov::intel_gpu::op::KVCache[l]` 管理和更新；`QSASDPA` 只消费
  已经更新好的主 K/V，绝不写入、扩容、搬迁或改变主 K/V cache。QSASDPA 不包含
  dense Q0；QSA paged path 的 Q0 继续由原有 QSA 实现负责。实际 dispatch 必须显式
  表达：

  ```text
  KVCache[l] completion -> qsa_inst[l] Q1/Q2/Q3
  qsa_inst[l] completion -> hidden/output of layer l
  ```

  `QSAIndexerCache[l]` 和 `QSASDPA[l]` 不能各自创建两份独立的 raw-K/summary
  runtime state。lowering 必须通过稳定的 layer/state identity 把图级
  `QSAIndexerCache[l]` 绑定到唯一的 `qsa_inst[l]` runtime owner；如果当前 graph
  仍保留两个独立 node，则它们只能共享同一个 runtime state handle，不能各自执行
  allocation、relocation 或 state publication。

  ### 15.3 每层 IndexerCache 的 state

  每个 `IndexerCache[l]` 至少包含两个独立 state：

  ```text
  raw-K state:
    memory handle / per-sequence base
    [B, raw_capacity, Di] storage
    raw_length[B]
    raw_capacity[B] 或统一 capacity

  summary state:
    memory handle / per-sequence base
    [B, summary_capacity, Di] storage
    summary_length[B]
    summary_capacity[B] 或统一 capacity

  common metadata:
    layer/state identifier
    sequence stride and base offsets (KVCache-style)
    position base
    block phase / absolute block origin
    last GPU-use completion event
    optional beam/trim metadata
  ```

  raw-K 与 summary 必须独立扩容，因为 raw-K 每个新 token 增加一项，而 summary
  只有压缩 block 完成时才增加一项。两者的 capacity、stride、trim 边界和 runtime
  shape 不能相互推导。

  `QSA_PA_BLOCK_SIZE` 不参与 dense cache capacity 计算。它只属于旧 paged backend
  的 page stride；dynamic dense backend 使用逻辑 block 大小 `R` 和 runtime stride。

  ### 15.4 职责边界

  #### `IndexerCache` 负责

  - 创建或获取 raw-K/summary 的 VariableState；
  - 首次调用时分配初始 memory；
  - 计算新 raw-K 和 summary 所需长度；
  - 检查 capacity，必要时分配新 memory；
  - 搬迁有效历史；
  - 发布 QSASDPA 可写的 cache descriptor；
  - 管理 shape、capacity、length、per-sequence base、memory handle 和最后一次 GPU
    使用完成状态；这些是 `qsa_inst` 内部 runtime state 的职责，OpenVINO op 本身
    只保存 graph-level Variable/config。
  - reset、trim、beam reorder 和 release。

  #### `IndexerCache` 不负责

  - indexer Q/K projection；
  - RMSNorm 或 RoPE；
  - raw-K 内容计算；
  - block mean pooling；
  - summary RMSNorm/RoPE；
  - Q2 score/top-k；
  - Q3 attention；
  - 独立 commit graph node；
  - QSASDPA 之后的 finalization kernel。

  #### `QSASDPA` 负责

  - indexer Q 计算；
  - 新 raw-K 计算与写入；
  - completed block summary 计算与写入；
  - Q2 score/top-k；
  - Q3 sparse attention；
  - 产生本层 completion event。

  ### 15.5 Reserve contract

  `IndexerCache[l]` 的 reserve 是同步 CPU manager 的 planning/allocation action，
  不是内容计算 kernel，也不是需要独立 GPU event 的 graph operation。输入包括：

  ```text
  old_raw_length[b]
  old_summary_length[b]
  past_lens[b]
  subsequence_begins[b + 1]
  new_token_count[b]
  compress_ratio = R
  last_qsa_completion[l]（仅在复用/搬迁仍被 GPU 使用的 storage 时需要等待）
  ```

  对每个 sequence 计算：

  ```text
  new_raw_length[b] = old_raw_length[b] + new_token_count[b]

  new_summary_length[b]
    = old_summary_length[b]
    + completed_blocks(old_raw_length[b], new_raw_length[b], R,
                       block_phase[b])

  raw_write_range[b]     = [old_raw_length[b], new_raw_length[b])
  summary_write_range[b] = [old_summary_length[b], new_summary_length[b])
  ```

  其中 `completed_blocks(old, new, R, block_phase)` 不是简单地假设每次调用从
  position 0 开始。无 trim 时可以退化为：

  ```text
  completed_blocks(old, new, R, 0) = floor(new / R) - floor(old / R)
  ```

  若发生 trim、滑窗或保留 partial block，则 descriptor 还必须携带当前 sequence
  的 `position_base`/`block_phase`，并按绝对 token 位置判断本次跨过了哪些 block
  boundary。这样才能保证 partial block 只生成一次 summary，且 trim 后不会重复或
  遗漏 summary。`old_summary_length` 必须与该 phase 对应，不能仅由 raw length
  重新猜测。

  reserve 需要完成：

    1. 在 CPU 侧完成本次 reserve；如果旧 storage 仍有未完成的 GPU 使用，则在复用或
      释放该 storage 前等待 `last_qsa_completion[l]`；如果 relocation 需要 device-to-
      device copy，则该 copy 必须在 reserve 返回前完成，或由同一个 runtime owner
      内部同步完成。不得把一个尚未完成的 relocation 隐藏在已经返回的 descriptor 后；
  2. 判断 raw-K/summary 是否需要扩容；
  3. 必要时分配新 memory 并搬迁 `[0, old_length)` 的有效历史；
  4. 生成新的 base、stride、capacity 和 write range；
  5. 发布本次 QSASDPA 使用的 descriptor；
  6. reserve 返回后直接把 descriptor 交给 QSASDPA；不产生 `reserve_event[l]`。

  reserve 不读取 hidden state，也不计算新的 indexer 内容。

  ### 15.6 Writable descriptor

  建议使用明确的 runtime descriptor 解耦 manager 和 QSASDPA：

  ```cpp
  struct qsa_indexer_cache_desc {
    device_ptr raw_k_base;
    device_ptr summary_base;
    uint32_t raw_k_stride;
    uint32_t summary_stride;

    // These are device-visible per-sequence metadata only when Q1/Q2/Q3
    // reads them in the kernel. The runtime owner may also keep host-side
    // scalar/array views for reserve and bookkeeping.
    device_ptr old_raw_length;
    device_ptr old_summary_length;
    device_ptr new_raw_length;
    device_ptr new_summary_length;
    device_ptr position_base;
    device_ptr block_phase;

    uint32_t raw_capacity;
    uint32_t summary_capacity;
    uint32_t indexer_dim;
    uint64_t reservation_id;
  };
  ```

  上述 `device_ptr` 是设计示意，不是要求最终 C++ API 把所有字段都实现成裸 GPU
  指针。它只表示“该字段如果作为 kernel 参数使用，必须指向 device-visible
  metadata”。`reserve` 在 CPU 侧仍可使用 host-side `uint32_t` scalar/array 或
  runtime memory abstraction；只有在 Q1/Q2/Q3 需要按 `b` 动态读取这些值时，才需要
  将对应数组放入 device-visible buffer，并在 descriptor 中传入其 device address。
  因此最终实现可以采用：

  ```text
  host state:   CPU-visible old/new length, phase and reservation bookkeeping
  device view:  optional device-visible length/phase arrays consumed by CM kernels
  ```

  两者必须来自同一次 reserve，不能分别维护互相矛盾的值。若某个 metadata 只在
  CPU 侧使用，则应保留为 host/runtime field，不应伪装成 kernel 参数。

  ### 15.7 QSASDPA 执行阶段

  ```text
  Q1  indexer projection + Q normalization/RoPE
    raw-K projection + bounded raw-K append
    completed-block pooling + summary append
  Q2  summary/raw-K score + top-k
  Q3  sparse attention over main K/V
  ```

  主 K/V update 不属于 QSASDPA 的任何 stage。进入 QSASDPA 前，调用方必须保证
  `KVCache[l]` 已完成本轮主 K/V 写入；QSASDPA 只读取该 cache。

  Q1 必须严格遵守 descriptor：

  ```text
  raw-K writes:
    raw_base[b] + [old_raw_length[b], new_raw_length[b])

  summary writes:
    summary_base[b] + [old_summary_length[b], new_summary_length[b])
  ```

  Q1 不申请 memory、不修改 base/stride/capacity，也不自行决定 logical length。

  ### 15.8 Partial block 规则

  1. 新 token 先全部追加到 raw-K state；
  2. block 未完成时只写 raw-K，不写 summary；
  3. 下一次调用继续追加缺失的 raw-K token；
  4. 当绝对 token position 跨过 `block_start + R` 时生成 exactly one summary；
  5. summary 写入位置由 `old_summary_length` 和 completed block 数决定；
  6. 已完成 block 不得重复生成 summary。

  summary 生成需要读取完整 raw-K block，包括上一轮保留的 partial raw-K 和本轮
  新写入的 raw-K。Q1 可以用 SLM/register 复用本轮数据，但跨调用历史必须从
  raw-K state 连续读取。

  ### 15.9 Dynamic dense storage 和连续寻址

  QSASDPA 复用现有 `cldnn::qsa` primitive，通过 JIT mode 选择 dense/paged 地址
  后端，不新增 `cldnn::qsa_dense` primitive。建议增加以下 JIT 控制：

  ```text
  QSA_SDPA_DENSE          = 1
  QSA_USE_RUNTIME_LENGTH  = 1
  QSA_RAW_K_STRIDE
  QSA_SUMMARY_STRIDE
  QSA_MAIN_K_STRIDE
  QSA_MAIN_V_STRIDE
  ```

  dense 地址逻辑为：

  ```text
  raw_k_ptr  = raw_base[b]     + position * raw_k_stride   + dim_offset
  summary_ptr= summary_base[b] + summary_slot * summary_stride + dim_offset
  main_ptr   = main_base[b]    + head/position runtime stride
  ```

  `raw_k_base[b]` 和 `summary_base[b]` 应复用现有 `KVCache` 的 per-sequence
  addressing 方式：manager 为每个 sequence 保存 base device pointer 或等价的
  memory offset，并配合 sequence stride/element stride 提供 descriptor。这里的
  `b` 不代表把所有 sequence 强行压成一个固定 page；它只是选择该 sequence 的
  独立 base。这样可以自然支持不同 sequence 的 valid length 和后续扩容，同时不
  改变现有 `KVCache` 的 state/lifecycle 设计原则。

  当 `QSA_SDPA_DENSE=0` 时保留现有 `block_indices -> physical page -> offset`；
  当 `QSA_SDPA_DENSE=1` 时禁止使用 paged `kv_cache_offset`、`raw_k_offset` 和
  `summary_offset` 作为动态 cache 地址。两种 backend 可以共享数学算法和 dispatch
  逻辑，但必须使用不同的 address helper；primitive、stage dispatch 和现有 QSA
  JIT 配置路径保持复用。

  ### 15.10 同步与 event contract

  `QSAIndexerCache` 的 OpenVINO op 只声明 state；类似 `KVCache` op 与
  `kv_cache_inst` 的关系，实际 runtime state 和 reserve 由执行 `cldnn::qsa` 的
  `qsa_inst` 持有。reserve 的 memory/state 部分在 CPU 侧同步完成，因此不需要把
  `IndexerCache` 伪装成一个异步 GPU producer。每层 `l` 的正常调用顺序是：

  ```text
  qsa_inst[l]
      -> CPU-side indexer_cache_runtime_state.reserve()
      - capacity/relocation/descriptor 完成
      - reserve 返回
      |
      v
  cldnn::qsa[l]
      - QSA_SDPA_DENSE=1 时执行 dense Q1/Q2/Q3
      |
      v
  qsa_event[l]
      |
      +--> next use of IndexerCache[l]
      +--> hidden_ready[l + 1]
  ```

  reserve 的同步返回值表示扩容/搬迁已经完成，descriptor 对同一个 `qsa_inst` 的
  QSA kernel 有效；不需要 `reserve_event[l]`。`qsa_event[l]` 只表示本层 Q1 写入
  和 Q2/Q3 读取已经完成，用于 GPU 后续使用、storage 复用以及 layer output 依赖。
  reserve 已精确计算写入数量和 capacity，logical length 的发布作为同一个
  `qsa_inst` runtime execution 的 state bookkeeping 完成，不是独立 graph node，
  也不需要写入后扫描、summary 校验或额外 finalization kernel。

  层间只通过 hidden/output 建立依赖：

  ```text
  qsa_event[l] -> hidden_ready[l + 1]
  hidden_ready[l + 1] -> QSASDPA[l + 1]
  ```

  下一层不读取上一层的 IndexerCache，因此不建立 `IndexerCache[l] ->
  IndexerCache[l + 1]` 的 cache 依赖。若 CPU manager 未来改为异步 GPU relocation，
  只需在那时为 relocation 增加内部同步，不改变 QSASDPA 的公开 contract。

  ### 15.11 Shape、capacity 和 valid length

  ```text
  capacity:
    实际分配的物理存储上限

  valid length:
    QSASDPA 可以读取或本次允许写入的逻辑边界

  runtime shape:
    对 graph/lowering 暴露的 cache 形状描述
  ```

  reserve 已知新的 valid length，因此可以更新 descriptor 和 runtime shape；物理
  capacity 可以大于 valid length。Q1/Q2/Q3 必须使用 valid length，不能扫描整个
  capacity，也不能从 static capacity shape 反推有效历史长度。state publication 必须
  以 descriptor 为单位完成：本次 descriptor 的 base、capacity、length、phase 和
  reservation id 必须来自同一次 reserve，不能让下一次调用观察到部分更新的 state。

  ### 15.12 生命周期操作

  - **Reset**：清空本层 raw-K/summary valid length、position base、last event 和
    reservation metadata；capacity 是否保留由统一 memory policy 决定。
  - **Trim**：manager 搬迁保留的 raw-K，按完整 block 边界裁剪 summary，更新
    position base 并发布新 descriptor；QSASDPA 不负责搬迁。
  - **Release**：释放前依赖本层最后一个 `qsa_event[l]`，确保所有 relocation、
    Q1 写入和 Q2/Q3 读取完成。
  - **Beam reorder**：同步重排本层 raw-K、summary 和主 K/V metadata；每层使用
    自己的 beam state。

  ### 15.13 推荐接口和实现拆分

  ```text
  ov::intel_gpu::op::QSAIndexerCache
    - declares raw-K Variable/state identity
    - declares summary Variable/state identity
    - stores graph-level QSA Config
    - analogous to ov::intel_gpu::op::KVCache

  cldnn::qsa
    - QSASDPA computation and cache writer
    - Q1/Q2/Q3 stages
    - dense or paged address backend selected by existing QSA JIT mode
    - owns or accesses qsa_indexer_cache_runtime_state

  qsa_inst::qsa_indexer_cache_runtime_state
    - runtime memory/capacity/length state
    - per-sequence base/stride and descriptor
    - reserve, relocation, reset, trim, release
    - analogous to kv_cache_inst's runtime VariableState handling
  ```

  不创建独立的 `cldnn::qsa_indexer_cache_state` primitive，也不创建独立的
  `QSAIndexerCache::commit`、`finalize` 或 `content_check` graph node。前者的职责
  融合为 `qsa_inst` 内部 runtime state/helper；后者已经由同步 reserve 返回值、
  descriptor、同一个 `qsa_inst` 的 state bookkeeping 和 GPU completion event 表达。

  ### 15.14 已确定的接口决策与剩余问题

  以下接口已经确定，不再作为开放问题：

  1. `QSAIndexerCache` 是图级 state declaration，声明 raw-K/summary 两个 Variable
     和 QSA Config；实际 runtime manager 融合到 `qsa_inst` 内部，采用
     `KVCache`/`kv_cache_inst` 的分层方式。
  2. per-sequence base、memory offset 和 stride 复用现有 `KVCache` 的对应抽象。
  3. 主 K/V 始终由已有 `KVCache` 更新，QSASDPA 不提供 Q0 update。
  4. `QSA_SDPA_DENSE` 复用现有 `cldnn::qsa` 的 JIT mode，不新增
     `cldnn::qsa_dense` primitive。
  5. 不创建独立的 `cldnn::qsa_indexer_cache_state`、`reserve_event` 或
     `QSAIndexerCache::commit` graph node。

    首期实现策略也在此处固定，避免把未实现能力误当成隐含 contract：

    1. runtime shape 对外暴露 valid length；physical capacity 和 explicit length/phase
      metadata 由 descriptor 管理，kernel 不能从 capacity shape 推断 valid length；
    2. 首期限定为单 batch/单序列 decode，trim 仅允许在完整 block 边界执行；beam
      reorder、ragged/continuous batching 和滑窗 trim 作为后续扩展，启用前必须保留
      `position_base`/`block_phase` 的同一套 descriptor 语义；
    3. device error 采用 invalidate-current-state 策略：失败 reservation 不得发布，
      已发布但 dispatch 失败的 state 标为 invalid，调用方必须显式 reset/recover；
      首期不提供 shadow-state 事务回滚。

  ### 15.15 `reserve` 与现有 `KVCache` 流程的关系

  `IndexerCache.reserve()` **在 state/memory 管理语义上复用 `KVCache` 的思路，
  但不应被理解为复刻一个独立的 `KVCache` graph node**。两者的共同点是：都由
  runtime instance 维护 Variable-like state，都根据历史 valid length 和本次 token
  数保证 capacity，都能使用 per-sequence base/stride，并在必要时完成扩容、搬迁和
  lifecycle 操作。

  现有 `KVCache` 的典型执行方式是：

  ```text
  caller provides current K/V as new_token_data
      -> kv_cache_inst reads current state
      -> ensure capacity / relocate if necessary
      -> append or copy new_token_data
      -> update state shape and valid length
      -> return the updated cache view
  ```

  `KVCache` 可以在一次 runtime execution 中同时完成规划、数据复制和 state 更新，
  因为它已经拿到了可以直接复制的 `new_token_data`。

  `IndexerCache` 的流程也由同一个 `qsa_inst` runtime owner 串联，只是其中 reserve
  是同步 CPU-side memory/state 管理，而 Q1/Q2/Q3 是 GPU kernel：

  ```text
  qsa_inst[l]
      -> CPU-side indexer_cache_runtime_state.reserve()
      -> 读取 raw-K/summary 的旧 valid length
      -> 计算本次 new length 和 raw/summary write range
      -> ensure capacity / relocate if necessary
      -> 发布 per-sequence base、stride、capacity、old/new length
      -> reserve 返回 writable descriptor

      -> cldnn::qsa (QSA_SDPA_DENSE=1)
      -> 计算 indexer Q 和新 raw-K
      -> 写入 reserve 授权的 raw-K range
      -> 计算并写入新完成 block 的 summary range
      -> 执行 Q2/Q3
      -> 产生 qsa_event[l]

      -> qsa_inst state bookkeeping
      -> 以 reserve 已确定的 new length 作为下一次调用的逻辑边界
      -> 复用 storage 前使用 qsa_event[l] 保证本次 GPU 写入/读取完成
  ```

  因此，调用顺序与 `KVCache` 的核心生命周期是一致的：

        | 阶段 | `KVCache` | `IndexerCache` |
        |---|---|---|
        | 读取历史 | 读取 Variable/state 的 valid length | 读取 raw-K/summary 各自的 valid length |
        | 规划容量 | 根据 `new_token_data` 和目标长度规划 | 根据 token 数、压缩比 `R` 和 block boundary 规划 |
        | 扩容搬迁 | manager/runtime 内部完成 | reserve 的同步 CPU manager 完成 |
        | 新内容 | 调用方已经提供 K/V | QSASDPA 在 GPU 上计算 raw-K/summary |
        | state 写入 | cache execute 内部完成 | reserve 发布 planned length，QSASDPA 只写 descriptor 授权范围 |
        | 完成依赖 | cache runtime 返回/事件保证可见 | `qsa_event[l]` 保证 GPU 写入和读取完成 |

  关键差异只有两点：

      1. `KVCache` 的输入内容已经存在，所以它可以把“容量管理”和“数据追加”放在
        同一个 runtime operation 中；`IndexerCache` 的 raw-K 和 summary 内容依赖
        QSASDPA 的 Q1 计算，因此 reserve 不能替代 QSASDPA，也不能自行生成内容。
      2. `IndexerCache` 的 summary 不是每个 token 都追加，而是只有 block 完成时才追加；
        所以 reserve 必须同时计算 raw-K range 和 summary range，不能只套用一个
        `new_token_data` 的单轴 append。

  结论是：**reserve 的 state/capacity/base/lifecycle 语义与 `KVCache` 一致，但数据
  生产者和调用边界不同**。当前设计不需要 `reserve_event`、独立 `commit` 或
  finalization node；同步 reserve 返回表示 descriptor 已准备好，同一个 `qsa_inst`
  内部完成 state bookkeeping，`qsa_event[l]` 负责 GPU 完成后的可见性、storage
  复用和后续执行依赖。

    ### 15.16 最终设计不变量和失败路径

    为避免实现时出现“图级 state、runtime state 和 kernel descriptor 各自维护一份
    真相”的问题，必须满足以下不变量：

    1. **单一 runtime owner**：每个 layer/state identity 只对应一个
      `qsa_inst::qsa_indexer_cache_runtime_state`；`QSAIndexerCache` graph node 和
      `QSASDPA` graph node（如果同时存在）只能引用它，不能各自分配 cache。
    2. **主 KV 只读**：`qsa_inst` 对主 `KVCache[l]` 只有 read dependency，没有 write、
      resize 或 relocation 入口；主 KV completion 必须先于 Q2/Q3。
    3. **descriptor 自洽**：base、stride、capacity、old/new length、position phase 和
      reservation id 必须来自同一个 reserve；Q1 只能写入 descriptor 授权的两个
      range。
    4. **状态单调性**：正常 decode 中 raw length 不下降，summary length 只在完整
      block 完成时增加；trim/reset 是显式 lifecycle 操作，不能由普通 Q1 调用隐式
      触发。
    5. **事件闭合**：旧 storage 的复用、释放或搬迁必须等待对应的 `qsa_event`；主
      KV 的读取必须等待 `KVCache` completion；layer output 必须发布
      `qsa_event` 后才能被下一层消费。
    6. **失败不发布半状态**：reserve 失败时不得替换当前 descriptor；relocation
      失败时不得更新 base/capacity；QSA kernel dispatch 失败时当前 reservation 必须
      标记 invalid，后续调用只能显式 reset/recover，不能把未完成的 new length 当成
      已提交历史继续使用。

    对应的最小验收场景是：首次调用、partial block、一次跨 block boundary、capacity
    growth/relocation、连续两次 infer、reset、trim 边界、主 KV event 缺失、Q1 dispatch
    失败和下一层 event 依赖。任何场景如果需要额外的 hidden commit、cache scan 或
    独立 reserve event，说明上述单一 owner 或 descriptor contract 没有正确落地。

  除上述接口选择外，最终职责和执行语义已经确定：

  ```text
  QSAIndexerCache[l]
    = graph-level Variable/config/state identity declaration

  qsa_inst[l]
    = KVCache-style runtime owner
      reserve/memory/capacity/shape state
      + indexer computation/cache writer/Q2/Q3 reader

  reserve
    = one-shot capacity and write-plan calculation

  CPU reserve return
    = descriptor/capacity readiness for QSASDPA

  completion event
    = GPU cache reuse/release and layer-output dependency

  commit/finalization/content-check
    = not required
  ```

