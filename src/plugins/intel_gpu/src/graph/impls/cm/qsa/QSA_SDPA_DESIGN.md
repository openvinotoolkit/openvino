## 15. Final Design Baseline: Layered QSASDPA + IndexerCache

  This chapter is the **final design baseline** for the current discussion. It consolidates
  the exploration, trade-offs, and corrections from the preceding sections into a design
  that can be used directly for human review, interface design, and implementation
  breakdown. If any historical section conflicts with this chapter, this chapter takes
  precedence.

  ### 15.1 Design Goals

  The final design satisfies the following requirements:

    1. Each transformer layer has its own `IndexerCache`; indexer cache memory or state is
      not shared between layers.
    2. `QSAIndexerCache` declares the raw-K/summary Variables and state identity at graph
      level. Its runtime state is owned by the `qsa_inst` that executes `cldnn::qsa`, which
      manages cache allocation, dynamic growth, relocation, shape/capacity/valid-length
      bookkeeping, reset, trim, beam metadata, and release. This layering is consistent
      with `KVCache`/`kv_cache_inst`.
    3. `IndexerCache` **does not compute new indexer cache content** and does not perform
      projection, RMSNorm, RoPE, raw-K pooling, or summary generation.
    4. `QSASDPA`/`cldnn::qsa` computes the new data and updates the cache for the current
      invocation: indexer Q, raw-K, completed-block summaries, and Q2/Q3 attention reads.
      Cache memory and logical state remain managed by runtime state inside the same
      `qsa_inst`.
    5. Before QSASDPA executes, `reserve` computes the new cache lengths, capacities, and
      write ranges in one step from runtime token/block information. No independent
      `commit` node or runtime finalization is required.
    6. The current paged `QSA` path remains unchanged. Dynamic `QSASDPA` uses contiguous
      dense caches with direct-stride addressing.
    7. `IndexerCache` reserve/growth/relocation is synchronous CPU-side management. Once
      reserve returns, the descriptor and write ranges required by the QSASDPA kernel are
      ready. No additional `IndexerCache`↔`QSASDPA` event is created for the CPU manager.
      The GPU side still retains the QSASDPA completion event for subsequent GPU use,
      reset/trim/release, and output dependencies; no cache-content checking phase is
      added.

  ### 15.2 Per-Layer Component Relationship

  ```text
               layer l

  main K/V projection -----------------> existing KVCache[l]
                         |
                         | dynamic dense main K/V
                         v
  hidden[l] -------------------------> QSASDPA[l]
                         ^
                         |
                  writable descriptor (synchronous reserve return)
                         |
                    IndexerCache[l]
                  raw-K state + summary state

  QSASDPA[l] outputs:
    attention output[l]
    hidden/output for layer l + 1
    qsa_event[l]

  qsa_event[l] -> hidden_ready[l + 1] -> QSASDPA[l + 1]
  ```

  Main K/V is always managed and updated by the existing
  `ov::intel_gpu::op::KVCache[l]`. `QSASDPA` only consumes the already-updated main
  K/V; it never writes, grows, relocates, or otherwise changes the main K/V cache.
  QSASDPA does not contain dense Q0; Q0 on the QSA paged path continues to be handled
  by the existing QSA implementation. The actual dispatch must explicitly express:

  ```text
  KVCache[l] completion -> qsa_inst[l] Q1/Q2/Q3
  qsa_inst[l] completion -> hidden/output of layer l
  ```

  `QSAIndexerCache[l]` and `QSASDPA[l]` must not create two independent copies of the
  raw-K/summary runtime state. Lowering must bind graph-level
  `QSAIndexerCache[l]` to the single `qsa_inst[l]` runtime owner through a stable
  layer/state identity. If the graph retains two separate nodes, they may only share the
  same runtime-state handle; neither node may independently perform allocation,
  relocation, or state publication.

  ### 15.3 Per-Layer IndexerCache State

  Each `IndexerCache[l]` contains at least two independent states:

  ```text
  raw-K state:
    memory handle / per-sequence base
    [B, raw_capacity, Di] storage
    raw_length[B]
    raw_capacity[B] or a shared capacity

  summary state:
    memory handle / per-sequence base
    [B, summary_capacity, Di] storage
    summary_length[B]
    summary_capacity[B] or a shared capacity

  common metadata:
    layer/state identifier
    sequence stride and base offsets (KVCache-style)
    position base
    block phase / absolute block origin
    last GPU-use completion event
    optional beam/trim metadata
  ```

  Raw-K and summary must grow independently because raw-K gains one entry per new token,
  while summary gains an entry only when a compressed block is completed. Their capacity,
  stride, trim boundaries, and runtime shapes must not be derived from one another.

  `QSA_PA_BLOCK_SIZE` does not participate in dense-cache capacity calculation. It belongs
  only to the page stride of the legacy paged backend; the dynamic dense backend uses the
  logical block size `R` and the runtime stride.

  ### 15.4 Responsibility Boundaries

  #### `IndexerCache` is responsible for

  - Creating or retrieving the raw-K/summary VariableState;
  - Allocating initial memory on the first invocation;
  - Computing the lengths required for new raw-K and summary entries;
  - Checking capacity and allocating new memory when necessary;
  - Relocating valid history;
  - Publishing a cache descriptor writable by QSASDPA;
  - Managing shape, capacity, length, per-sequence base, memory handle, and the completion
    state of the last GPU use. These are responsibilities of runtime state inside
    `qsa_inst`; the OpenVINO op itself stores only the graph-level Variable/config.
  - Reset, trim, beam reorder, and release.

  #### `IndexerCache` is not responsible for

  - Indexer Q/K projection;
  - RMSNorm or RoPE;
  - Raw-K content computation;
  - Block mean pooling;
  - Summary RMSNorm/RoPE;
  - Q2 scoring/top-k;
  - Q3 attention;
  - An independent commit graph node;
  - A finalization kernel after QSASDPA.

  #### `QSASDPA` is responsible for

  - Computing indexer Q;
  - Computing and writing new raw-K;
  - Computing and writing summaries for completed blocks;
  - Q2 scoring/top-k;
  - Q3 sparse attention;
  - Producing the completion event for the layer.

  ### 15.5 Reserve Contract

  `IndexerCache[l]` reserve is a synchronous CPU-manager planning/allocation action. It
  is neither a content-computation kernel nor a graph operation requiring an independent
  GPU event. Its inputs are:

  ```text
  old_raw_length[b]
  old_summary_length[b]
  past_lens[b]
  subsequence_begins[b + 1]
  new_token_count[b]
  compress_ratio = R
  last_qsa_completion[l] (wait only when reused/relocated storage is still in GPU use)
  ```

  For each sequence, compute:

  ```text
  new_raw_length[b] = old_raw_length[b] + new_token_count[b]

  new_summary_length[b]
    = old_summary_length[b]
    + completed_blocks(old_raw_length[b], new_raw_length[b], R,
                       block_phase[b])

  raw_write_range[b]     = [old_raw_length[b], new_raw_length[b])
  summary_write_range[b] = [old_summary_length[b], new_summary_length[b])
  ```

  `completed_blocks(old, new, R, block_phase)` must not simply assume that every invocation
  starts at position 0. Without trim, it can be reduced to:

  ```text
  completed_blocks(old, new, R, 0) = floor(new / R) - floor(old / R)
  ```

  If trim, sliding windows, or retained partial blocks are used, the descriptor must also
  carry the current sequence's `position_base`/`block_phase` and determine which block
  boundaries were crossed this time using absolute token positions. This ensures that a
  partial block produces exactly one summary and that trim does not duplicate or omit a
  summary. `old_summary_length` must correspond to that phase and cannot be reconstructed
  from raw length alone.

  Reserve must:

    1. Complete the reserve on the CPU side. If old storage still has unfinished GPU use,
      wait for `last_qsa_completion[l]` before reusing or releasing it. If relocation
      requires a device-to-device copy, complete that copy before reserve returns, or
      synchronize it internally within the same runtime owner. An unfinished relocation
      must not be hidden behind an already-returned descriptor;
  2. Determine whether raw-K/summary capacity must grow;
  3. Allocate new memory when necessary and relocate valid history in `[0, old_length)`;
  4. Generate the new base, stride, capacity, and write ranges;
  5. Publish the descriptor used by this QSASDPA invocation;
  6. Pass the descriptor directly to QSASDPA after reserve returns; do not create a
     `reserve_event[l]`.

  Reserve does not read hidden state or compute new indexer content.

  ### 15.6 Writable Descriptor

  A dedicated runtime descriptor is recommended to decouple the manager from QSASDPA:

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

  The `device_ptr` fields above are design notation; the final C++ API is not required to
  represent every field as a raw GPU pointer. They mean that, if a field is used as a
  kernel argument, it must point to device-visible metadata. `reserve` may still use
  host-side `uint32_t` scalars/arrays or a runtime memory abstraction. Only when Q1/Q2/Q3
  need to read a value dynamically by `b` must the corresponding array be placed in a
  device-visible buffer and its device address passed through the descriptor. The final
  implementation may therefore use:

  ```text
  host state:   CPU-visible old/new length, phase and reservation bookkeeping
  device view:  optional device-visible length/phase arrays consumed by CM kernels
  ```

  Both views must come from the same reserve and must not maintain contradictory values.
  If metadata is used only on the CPU side, it should remain a host/runtime field rather
  than being disguised as a kernel argument.

  ### 15.7 QSASDPA Execution Stages

  ```text
  Q1  indexer projection + Q normalization/RoPE
    raw-K projection + bounded raw-K append
    completed-block pooling + summary append
  Q2  summary/raw-K score + top-k
  Q3  sparse attention over main K/V
  ```

  Main K/V update is not part of any QSASDPA stage. Before entering QSASDPA, the caller
  must ensure that `KVCache[l]` has completed the main K/V write for the current
  invocation; QSASDPA only reads that cache.

  Q1 must strictly follow the descriptor:

  ```text
  raw-K writes:
    raw_base[b] + [old_raw_length[b], new_raw_length[b])

  summary writes:
    summary_base[b] + [old_summary_length[b], new_summary_length[b])
  ```

  Q1 does not allocate memory, modify base/stride/capacity, or decide the logical length
  on its own.

  ### 15.8 Partial-Block Rules

    1. Append all new tokens to the raw-K state first;
    2. While a block is incomplete, write only raw-K and do not write a summary;
    3. Continue appending missing raw-K tokens on the next invocation;
    4. Generate exactly one summary when the absolute token position crosses
      `block_start + R`;
    5. Determine the summary write position from `old_summary_length` and the number of
      completed blocks;
    6. Never generate a summary more than once for a completed block.

  Summary generation must read the complete raw-K block, including the partial raw-K
  retained from the previous invocation and the raw-K written in the current invocation.
  Q1 may reuse current-invocation data in SLM/registers, but history across invocations
  must be read continuously from the raw-K state.

  ### 15.9 Dynamic Dense Storage and Contiguous Addressing

  QSASDPA reuses the existing `cldnn::qsa` primitive and selects the dense/paged address
  backend through JIT mode; it does not add a `cldnn::qsa_dense` primitive. The following
  JIT controls are recommended:

  ```text
  QSA_SDPA_DENSE          = 1
  QSA_USE_RUNTIME_LENGTH  = 1
  QSA_RAW_K_STRIDE
  QSA_SUMMARY_STRIDE
  QSA_MAIN_K_STRIDE
  QSA_MAIN_V_STRIDE
  ```

  Dense address logic is:

  ```text
  raw_k_ptr  = raw_base[b]     + position * raw_k_stride   + dim_offset
  summary_ptr= summary_base[b] + summary_slot * summary_stride + dim_offset
  main_ptr   = main_base[b]    + head/position runtime stride
  ```

  `raw_k_base[b]` and `summary_base[b]` should reuse the per-sequence addressing approach
  of the existing `KVCache`: the manager stores a base device pointer or equivalent
  memory offset for each sequence and provides it in the descriptor together with the
  sequence and element strides. Here `b` does not mean that all sequences are forced
  into one fixed page; it selects the independent base for that sequence. This naturally
  supports different valid lengths and later growth without changing the existing
  `KVCache` state/lifecycle principles.

  When `QSA_SDPA_DENSE=0`, retain the existing
  `block_indices -> physical page -> offset` path. When `QSA_SDPA_DENSE=1`, do not use
  paged `kv_cache_offset`, `raw_k_offset`, or `summary_offset` as dynamic cache
  addresses. The two backends may share mathematical algorithms and dispatch logic, but
  must use different address helpers. The primitive, stage dispatch, and existing QSA
  JIT configuration path remain shared.

  ### 15.10 Synchronization and Event Contract

  The OpenVINO `QSAIndexerCache` op only declares state. As with the relationship between
  `KVCache` and `kv_cache_inst`, the actual runtime state and reserve are owned by the
  `qsa_inst` executing `cldnn::qsa`. The memory/state portion of reserve is completed
  synchronously on the CPU, so `IndexerCache` need not be disguised as an asynchronous
  GPU producer. The normal invocation order for layer `l` is:

  ```text
  qsa_inst[l]
      -> CPU-side indexer_cache_runtime_state.reserve()
      - capacity/relocation/descriptor ready
      - reserve returns
      |
      v
  cldnn::qsa[l]
      - execute dense Q1/Q2/Q3 when QSA_SDPA_DENSE=1
      |
      v
  qsa_event[l]
      |
      +--> next use of IndexerCache[l]
      +--> hidden_ready[l + 1]
  ```

  The synchronous return from reserve means that growth/relocation is complete and that
  the descriptor is valid for the QSA kernel of the same `qsa_inst`; no
  `reserve_event[l]` is needed. `qsa_event[l]` only indicates that this layer's Q1 writes
  and Q2/Q3 reads have completed, and is used for later GPU use, storage reuse, and layer
  output dependencies. Because reserve has precisely computed the write counts and
  capacity, logical-length publication is runtime bookkeeping in the same `qsa_inst`
  execution, not an independent graph node. No post-write scan, summary validation, or
  additional finalization kernel is required.

  Layers establish dependencies only through hidden/output:

  ```text
  qsa_event[l] -> hidden_ready[l + 1]
  hidden_ready[l + 1] -> QSASDPA[l + 1]
  ```

  The next layer does not read the previous layer's IndexerCache, so no
  `IndexerCache[l] -> IndexerCache[l + 1]` cache dependency is created. If the CPU
  manager later changes to asynchronous GPU relocation, only internal relocation
  synchronization needs to be added; the public QSASDPA contract does not change.

  ### 15.11 Shape, Capacity, and Valid Length

  ```text
  capacity:
    physical storage limit actually allocated

  valid length:
    logical boundary that QSASDPA may read or write in this invocation

  runtime shape:
    cache shape description exposed to graph/lowering
  ```

  Reserve knows the new valid length and can therefore update the descriptor and runtime
  shape; physical capacity may exceed valid length. Q1/Q2/Q3 must use valid length and
  must not scan the entire capacity or infer valid history length from a static capacity
  shape. State publication must be atomic at descriptor granularity: the base, capacity,
  length, phase, and reservation id in a descriptor must come from the same reserve, and
  the next invocation must never observe partially updated state.

  ### 15.12 Lifecycle Operations

  - **Reset**: Clear the current layer's raw-K/summary valid lengths, position base, last
    event, and reservation metadata. Whether capacity is retained is determined by a
    common memory policy.
  - **Trim**: The manager relocates retained raw-K, trims summary at complete-block
    boundaries, updates position base, and publishes a new descriptor. QSASDPA does not
    perform relocation.
  - **Release**: Before release, depend on the layer's last `qsa_event[l]` to ensure that
    all relocation, Q1 writes, and Q2/Q3 reads have completed.
  - **Beam reorder**: Synchronously reorder the layer's raw-K, summary, and main-K/V
    metadata. Each layer uses its own beam state.

  ### 15.13 Recommended Interface and Implementation Breakdown

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

  Do not create an independent `cldnn::qsa_indexer_cache_state` primitive or an
  independent `QSAIndexerCache::commit`, `finalize`, or `content_check` graph node. The
  former responsibility is folded into runtime state/helpers inside `qsa_inst`; the
  latter is expressed by the synchronous reserve return, descriptor, state bookkeeping
  in the same `qsa_inst`, and the GPU completion event.

  ### 15.14 Settled Interface Decisions and Remaining Issues

  The following interface decisions are settled and are no longer open questions:

    1. `QSAIndexerCache` is a graph-level state declaration containing two Variables for
      raw-K/summary and the QSA Config. The actual runtime manager is integrated into
      `qsa_inst` using the `KVCache`/`kv_cache_inst` layering model.
    2. Per-sequence base, memory offset, and stride reuse the corresponding abstractions
      from the existing `KVCache`.
    3. Main K/V is always updated by the existing `KVCache`; QSASDPA does not provide Q0
      update.
    4. `QSA_SDPA_DENSE` reuses the existing `cldnn::qsa` JIT mode and does not add a
      `cldnn::qsa_dense` primitive.
    5. Do not create an independent `cldnn::qsa_indexer_cache_state`, `reserve_event`, or
      `QSAIndexerCache::commit` graph node.

    The initial implementation strategy is also fixed here, so that unimplemented
    capabilities are not mistaken for an implicit contract:

    1. Runtime shape exposes valid length. Physical capacity and explicit length/phase
      metadata are managed by the descriptor; kernels must not infer valid length from
      capacity shape.
    2. The initial version is limited to single-batch/single-sequence decode, and trim is
      allowed only at complete-block boundaries. Beam reorder, ragged/continuous
      batching, and sliding-window trim are future extensions; before enabling them, the
      same `position_base`/`block_phase` descriptor semantics must be retained.
    3. Device errors use an invalidate-current-state policy: a failed reservation must not
      be published; state whose dispatch fails after publication is marked invalid, and
      the caller must explicitly reset/recover. The initial version does not provide
      shadow-state transactional rollback.

  ### 15.15 Relationship Between `reserve` and the Existing `KVCache` Flow

  `IndexerCache.reserve()` **reuses the `KVCache` approach for state/memory management,
  but must not be understood as replicating an independent `KVCache` graph node**. The
  common properties are that both maintain Variable-like state in a runtime instance,
  ensure capacity from historical valid length and the current token count, support
  per-sequence base/stride, and perform growth, relocation, and lifecycle operations when
  necessary.

  The typical execution flow of the existing `KVCache` is:

  ```text
  caller provides current K/V as new_token_data
      -> kv_cache_inst reads current state
      -> ensure capacity / relocate if necessary
      -> append or copy new_token_data
      -> update state shape and valid length
      -> return the updated cache view
  ```

  `KVCache` can complete planning, data copying, and state updates in one runtime
  execution because it already has `new_token_data` that can be copied directly.

  The `IndexerCache` flow is likewise coordinated by the same `qsa_inst` runtime owner,
  except that reserve is synchronous CPU-side memory/state management while Q1/Q2/Q3 are
  GPU kernels:

  ```text
  qsa_inst[l]
      -> CPU-side indexer_cache_runtime_state.reserve()
      -> read old raw-K/summary valid lengths
      -> compute this invocation's new lengths and raw/summary write ranges
      -> ensure capacity / relocate if necessary
      -> publish per-sequence base, stride, capacity, and old/new lengths
      -> reserve returns a writable descriptor

      -> cldnn::qsa (QSA_SDPA_DENSE=1)
      -> compute indexer Q and new raw-K
      -> write the raw-K range authorized by reserve
      -> compute and write the summary range for newly completed blocks
      -> execute Q2/Q3
      -> produce qsa_event[l]

      -> qsa_inst state bookkeeping
      -> use the new length determined by reserve as the next invocation's logical boundary
      -> use qsa_event[l] before reusing storage to ensure this invocation's GPU reads/writes complete
  ```

  Therefore, the invocation order is consistent with the core `KVCache` lifecycle:

        | Stage | `KVCache` | `IndexerCache` |
        |---|---|---|
        | Read history | Read Variable/state valid length | Read independent raw-K/summary valid lengths |
        | Plan capacity | Plan from `new_token_data` and target length | Plan from token count, compression ratio `R`, and block boundaries |
        | Growth/relocation | Completed inside manager/runtime | Completed by the synchronous CPU manager in reserve |
        | New content | K/V is already provided by the caller | Raw-K/summary are computed by QSASDPA on the GPU |
        | State write | Completed inside cache execution | Reserve publishes the planned length; QSASDPA writes only the ranges authorized by the descriptor |
        | Completion dependency | Cache runtime return/event guarantees visibility | `qsa_event[l]` guarantees completion of GPU writes/reads |

  There are only two key differences:

      1. `KVCache` already has the input content, so it can place capacity management and
        data append in one runtime operation. `IndexerCache` raw-K and summary content
        depends on QSASDPA's Q1 computation, so reserve cannot replace QSASDPA or generate
        content by itself.
      2. `IndexerCache` summary is not appended for every token; it is appended only when
        a block is complete. Therefore reserve must compute both raw-K and summary ranges,
        rather than applying a single-axis `new_token_data` append.

  The conclusion is: **reserve has the same state/capacity/base/lifecycle semantics as
  `KVCache`, but the data producer and invocation boundary are different**. The current
  design does not need a `reserve_event`, independent `commit`, or finalization node. A
  synchronous reserve return means that the descriptor is ready; state bookkeeping is
  completed inside the same `qsa_inst`, while `qsa_event[l]` handles post-GPU-completion
  visibility, storage reuse, and subsequent execution dependencies.

    ### 15.16 Final-Design Invariants and Failure Paths

    To prevent graph-level state, runtime state, and the kernel descriptor from each
    maintaining a separate source of truth, the following invariants must hold:

    1. **Single runtime owner**: Each layer/state identity maps to exactly one
      `qsa_inst::qsa_indexer_cache_runtime_state`. The `QSAIndexerCache` graph node and
      the `QSASDPA` graph node, if both exist, may only reference it; neither may allocate
      its own cache.
    2. **Main KV is read-only**: `qsa_inst` has only a read dependency on main
      `KVCache[l]`; it has no write, resize, or relocation entry point. Main-KV completion
      must precede Q2/Q3.
    3. **Self-consistent descriptor**: Base, stride, capacity, old/new length, position
      phase, and reservation id must come from the same reserve. Q1 may write only to the
      two ranges authorized by the descriptor.
    4. **Monotonic state**: During normal decode, raw length never decreases and summary
      length increases only when a complete block is finished. Trim/reset are explicit
      lifecycle operations and must not be triggered implicitly by an ordinary Q1 call.
    5. **Closed event dependencies**: Reuse, release, or relocation of old storage must
      wait for the corresponding `qsa_event`. Main-KV reads must wait for KVCache
      completion. Layer output may be consumed by the next layer only after
      `qsa_event` is published.
    6. **Do not publish partial state on failure**: A failed reserve must not replace the
      current descriptor; a failed relocation must not update base/capacity; and a failed
      QSA kernel dispatch must mark the current reservation invalid. Later calls may only
      proceed after explicit reset/recovery and must not treat an unfinished new length as
      committed history.

    The corresponding minimum acceptance scenarios are: first invocation, partial block,
    one block-boundary crossing, capacity growth/relocation, two consecutive inferences,
    reset, trim boundaries, missing main-KV event, Q1 dispatch failure, and next-layer
    event dependencies. If any scenario requires a hidden commit, cache scan, or
    independent reserve event, the single-owner or descriptor contract above has not
    been implemented correctly.

  Apart from the interface choices above, the final responsibilities and execution
  semantics are fixed:

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

