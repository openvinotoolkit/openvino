.. {#openvino_docs_ops_internal_RecurrentConvCacheUpdate}

RecurrentConvCacheUpdate
========================

**Versioned name**: *RecurrentConvCacheUpdate*

**Category**: *Internal*

Added for Qwen3.5/Qwen3.6 (GGML ``qwen35`` and ``qwen35moe``) in llama.cpp's
OpenVINO GPU backend. This independent operation combines depthwise causal
convolution with generic ``cache_r`` reset, reorder, and history writeback.
It supports in-place execution without changing the existing convolution operation.

**Inputs (zero-based)**

* 0 cache: ``f16/f32 [1,1,S,C*H]``, with per-row layout ``[channel,history]``.
* 1 active_rows and 2 extra_rows: independently ``i32/i64 [1,1,1,N]``.
* 3 reset_begin and 4 reset_length: independently ``i32/i64 [1,1,1,1]``.
* 5 extra_offsets: ``i32/i64 [1,1,1,2]``, ``(destination_begin,source_begin)``.
* 6 QKV: ``f16/f32 [1,B,C,T]``, where ``B`` equals the active-row count.
* 7 windows: ``i32/i64 [1,1,W,2]``, ordered ``(start_column,destination_row)`` pairs.
* 8 depthwise weights: ``f16/f32 [1,1,C,H+1]`` (required).

**Attribute**: positive ``history=H``.

**Outputs**

* 0 convolution ``[1,B,T,C]`` in QKV precision, accumulated in FP32.

**Semantics**

Logically reset source rows in ``[reset_begin,reset_begin+reset_length)`` before
reading either row map. Active histories are gathered from that original source,
then concatenated with QKV for convolution. The selected suffix
``extra_rows[source_begin:]`` reorders rows to consecutive destinations beginning
at ``destination_begin``. Each window copies ``H`` columns of the concatenated
history/QKV to ``B`` cache rows. Valid window starts are in ``[0,T]``. Reset,
reorder, and window writes occur in that order; later windows win overlaps.
The caller supplies in-bounds maps, ranges, and windows. Runtime dimensions and
metadata may change on one compiled model.

Cache reads, retained rows, and history writes round through QKV precision,
matching the prior materialized computation path. Cache input 0 is updated in place;
no cache output is allocated.

**Writable cache input and GPU ordering**

Cache input 0 is updated in place. The caller binds a writable GPU RemoteTensor
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
