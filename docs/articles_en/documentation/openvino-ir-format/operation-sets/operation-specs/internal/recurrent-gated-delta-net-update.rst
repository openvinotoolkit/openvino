.. {#openvino_docs_ops_internal_RecurrentGatedDeltaNetUpdate}

RecurrentGatedDeltaNetUpdate
============================

**Versioned name**: *RecurrentGatedDeltaNetUpdate*

**Category**: *Internal*

Added for Qwen3.5/Qwen3.6 (GGML ``qwen35`` and ``qwen35moe``) in llama.cpp's
OpenVINO GPU backend. This independent operation combines Gated Delta Net
computation with generic ``cache_s`` reset, reorder, and MTP snapshot writeback.
It supports in-place execution without changing existing GDN operation contracts.

**Inputs (zero-based)**

* 0 query and 1 key: ``f16/f32 [B,T,H_k,K]``.
* 2 value: query precision, ``[B,T,H_v,V]``.
* 3 original cache: ``f16/f32 [1,1,S,H_v*V*K]``, per-row ``[head,value,key]`` layout.
* 4 gate and 5 beta: query precision, ``[B,T,H_v]``.
* 6 active_rows and 7 extra_rows: independently ``i32/i64 [1,1,1,N]``.
* 8 reset_begin and 9 reset_length: independently ``i32/i64 [1,1,1,1]``.
* 10 extra_offsets: ``i32/i64 [1,1,1,2]``, ``(destination_begin,source_begin)``.
* 11 write_control: ``i32/i64 [1,1,1,4]``,
  ``(destination_begin,sequence_count,snapshot_stride,source_begin)``.

**Attributes**

``fuse_qk_l2norm=false``, ``q_l2_norm_eps=1e-6``, and ``k_l2_norm_eps=1e-6``
have the computation meanings of :doc:`gated-delta-net`. Positive
``snapshot_count=1`` controls the newest-first states retained by this operation;
it does not change the ordinary GatedDeltaNet contract. Rank and head/feature dimensions are known at GPU
compilation; token/sequence/cache-row dimensions may change. All runtime
maps, ranges, and destinations must be in bounds.

**Outputs and semantics**

* 0 attention: query precision, ``[B,T,H_v,V]``.

Reset source rows logically before gather. Active states read the original
source before reorder. The extra-row suffix resets/reorders retained cache rows.
GDN computes all token outputs, accumulating in FP32. The newest
``min(snapshot_count,T)`` token-prefix states are written directly into cache,
newest first; no standalone snapshot output is allocated. ``snapshot_count=1``
writes the final state. With count greater than one, entry 0 still represents
the final state, and older entries support llama.cpp MTP rollback.

Selected snapshot source row ``source_begin+r`` writes destination
``destination_begin+(r/sequence_count)*snapshot_stride+r%sequence_count``.
Later source rows win overlaps, including zero snapshot stride. State reads,
retained rows, and snapshot writes round through query precision, then cache
precision, matching the former materialized computation path.

**Writable cache input and GPU ordering**

Cache input 3 is updated in place. The caller binds a writable GPU RemoteTensor
(device USM or host USM) and reads the updated cache after inference completes.
The cache must originate from a Parameter, optionally through exclusive
memory-preserving Reshapes. Other cache readers in the same inference and
computed temporary cache banks are rejected during GPU compilation. Separate
inputs and the computation output must not overlap the writable cache allocation. Ordinary host tensor
inputs are copied to GPU storage and do not provide caller-visible writeback.
The precision pass preserves the cache and its Reshape aliases in their original
precision to avoid a full-cache conversion before the operation.

Required active and nonidentity reorder sources are captured in private device
scratch before the compute/writeback kernel starts. The explicit dependency
between these stages protects cycles, duplicate sources and overlapping writes.
Identity reorder rows need no scratch writes. Each cache element has one writer;
already rounded retained values need no store. Runtime token/sequence counts
remain shape-agnostic and compiled-model export/import remains supported.

There is no cache Result, output binding, or immutable-input fallback within this
operation. Callers requiring immutable cache inputs must use an ordinary
functional decomposition instead. A failed inference can leave the cache partly
updated; clear or restore it before retrying.
