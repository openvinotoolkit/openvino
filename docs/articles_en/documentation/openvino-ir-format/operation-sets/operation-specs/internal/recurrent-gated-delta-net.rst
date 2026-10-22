.. {#openvino_docs_ops_internal_RecurrentGatedDeltaNet}

RecurrentGatedDeltaNet
=====================

**Versioned name**: *RecurrentGatedDeltaNet*

**Category**: *Internal*

Introduced for Qwen3.5/Qwen3.6 (GGML ``qwen35``/``qwen35moe``) in llama.cpp's
OpenVINO backend. It combines generic ``cache_s`` reset/reorder, GatedDeltaNet
computation, and ordered MTP snapshot writeback. One native GPU kernel replaces
active-state gather, GDN, and full-cache update, avoiding their intermediate
state tensors and full-cache precision conversions. The six-input
:doc:`gated-delta-net` contract remains unchanged.

**Inputs (zero-based)**

* 0 query: floating ``[B,T,H_k,K]``.
* 1 key: floating ``[B,T,H_k,K]``.
* 2 value: floating ``[B,T,H_v,V]``.
* 3 original cache: ``f16/f32 [1,1,S,H_v*V*K]``, each row laid out ``[head,value,key]``.
* 4 gate and 5 beta: floating ``[B,T,H_v]``.
* 6 active_rows, 7 extra_rows, 8 reset_begin, 9 reset_length, 10 extra_offsets:
  independent ``i32/i64`` tensors with the :doc:`recurrent-state-gather` contract.
* 11 write_control: ``i32/i64 [1,1,1,4]`` with the :doc:`recurrent-state-update` contract.

Q/K/V/gate/beta support the GPU GDN FP16/FP32 path. Shapes are dynamic with known
rank; all runtime ranges must be in bounds. The original cache remains immutable,
including cyclic row maps and duplicate source rows. Kernel allocations must be
distinct; the runtime may stage aliased remote input/output buffers.

**Attributes**

``fuse_qk_l2norm``, ``q_l2_norm_eps``, ``k_l2_norm_eps``, and positive
``snapshot_count`` have the :doc:`gated-delta-net` meanings.

**Outputs and semantics**

* 0 attention: ``[B,T,H_v,V]`` in query precision.
* 1 updated cache: ``[1,1,S,H_v*V*K]`` in cache precision.

Active states read the logically reset original cache before reorder. Computation
uses the usual GDN recurrence. The newest ``min(snapshot_count,T)`` states are
written directly into cache rows according to write_control, with later source
rows winning overlaps. Inactive rows retain their logically reset/reordered
values. No standalone snapshots output is allocated. Rejected MTP suffixes can
be restored from the retained cache rows through llama.cpp's existing row maps.

State reads and writes round through query precision, as do preserved cache rows,
matching the former materialized GPU precision path. GDN accumulation remains
FP32. The GPU precision pass preserves cache I/O precision while allowing normal
query/K/V computation precision selection.

GPU cache output staging uses device USM when device allocations are available and
no CPU consumer requires lockable memory. Caller input/output aliasing still needs
immutable kernel input storage; internal device staging and a final device copy
are permitted. This allocation policy does not imply zero-copy execution.
