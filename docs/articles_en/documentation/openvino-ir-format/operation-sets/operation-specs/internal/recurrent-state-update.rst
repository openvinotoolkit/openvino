.. {#openvino_docs_ops_internal_RecurrentStateUpdate}

RecurrentStateUpdate
====================

**Versioned name**: *RecurrentStateUpdate*

**Category**: *Internal*

**Short description**: Write active recurrent states or snapshot planes into a
full cache while preserving inactive slots and ordered overlapping writes.

These internal operations fuse recurrent-cache data movement used by hybrid
language models with convolution history and matrix state. They do not perform
attention or convolution arithmetic. Runtime row maps support concurrent sequences,
slot reuse/reset, and speculative-decoding snapshots without requiring a special
single-sequence path. Native implementations are provided by the Intel GPU plugin.

All inputs have rank four. Floating-point inputs support ``f16`` and ``f32``;
metadata inputs support ``i32`` and ``i64`` independently. ``S`` is the cache slot
count, ``F`` the flattened per-slot size, and ``B`` the active sequence count.
Dimensions may be dynamic, but rank must be known. Runtime metadata values must
satisfy the bounds below; the kernel does not validate out-of-range values.

The operation does not modify its input cache. Its GPU kernel reads immutable
source allocations, so arbitrary row-map cycles and repeated sources are safe.
The plugin can stage a caller's shared remote input/output tensor to preserve this
contract; kernel input/output allocations themselves must be distinct.

**Attributes**

``key_size`` and ``value_size`` default to zero. Both must be zero (no transpose)
or positive with a product dividing ``F``. Positive values convert source
``[head,key,value]`` matrices to cache ``[head,value,key]`` layout.

**Inputs (zero-based C++ port indices)**

* ``0 cache``: floating-point ``[1,1,S,F]``; normally the prepared output of
  :doc:`recurrent-state-gather`.
* ``1 states``: floating-point ``[1,1,N,F]`` containing flattened final states
  or snapshot planes in plane-major, sequence-minor order.
* ``2 control``: integer ``[1,1,1,4]`` containing
  ``(destination_begin, sequence_count, snapshot_stride, source_begin)``.

``sequence_count`` is positive, ``snapshot_stride`` nonnegative, and
``source_begin`` lies in ``[0,N]``. All selected destination rows must lie in
``[0,S)``. ``source_begin=N`` performs no writes. A partial final plane is allowed.

**Output**

``0 updated_cache``: ``[1,1,S,F]`` in cache precision.

**Reference semantics**

.. code-block:: python

   updated = cache[0, 0].copy()
   selected = states[0, 0, source_begin:]
   if key_size > 0:
       selected = selected.reshape(-1, F // (key_size * value_size), key_size, value_size)
       selected = selected.transpose(0, 1, 3, 2).reshape(-1, F)
   for r, state in enumerate(selected):
       plane, sequence = divmod(r, sequence_count)
       destination = destination_begin + plane * snapshot_stride + sequence
       updated[destination] = state.astype(cache.dtype)

Later source rows win if destination ranges overlap, including when
``snapshot_stride=0``. With ``sequence_count=2``, ``snapshot_stride=4``, and
``destination_begin=1``, six source rows write cache rows ``[1,2,5,6,9,10]``.
Source snapshots from :doc:`gated-delta-net` are newest first; this operation
preserves that order and leaves slot allocation to its caller.
