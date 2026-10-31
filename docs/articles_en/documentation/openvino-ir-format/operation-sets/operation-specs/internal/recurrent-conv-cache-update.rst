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
* 1 the complete updated cache: same shape and precision as input 0.

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
matching the prior materialized computation path. Output 1 is a full cache even
when only a few rows change.

**GPU allocation and ordering**

Distinct caller allocations retain immutable input semantics. For dynamic
execution, identical input/output device-USM or host-USM allocations can be bound directly
when the cache has an exclusive graph reader and no other input overlaps the
output. Memory-identical optimized Reshapes/views may connect the cache Parameter
to the operation; the entire alias chain must be exclusive, including the root
Parameter. The precision pass also preserves FP32 cache Reshape aliases when
compute uses FP16, avoiding full-cache precision reorders before the operation.
An independent reader at any step disables direct alias execution.
Required active and nonidentity reorder sources are first captured in
private device scratch. Scratch capacity is conservatively bounded by the
active-map plus full extra-map lengths; skipping identity rows avoids transfers
but does not reduce that allocated capacity. A dependency boundary completes this gather before the
compute/writeback kernel starts. Cycles and duplicate sources are therefore safe;
identity suffixes need no scratch writes. Each destination element has one writer.
Already rounded retained values need no store, but are still read for precision
rounding. There is no full-cache Result copy on this eligible alias path.

Other bindings, shared graph readers, incompatible layouts, static models, and
partial or unrelated overlaps use immutable-source output storage and any needed
Result copy. GPU command replay is currently disabled for this new operation
because its stage selection depends on runtime input/output aliasing.

An inference failure on the direct alias path can leave the caller's cache partly
updated. The caller must clear or restore the cache before retrying.
