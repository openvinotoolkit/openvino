.. {#openvino_docs_ops_internal_RecurrentStateGather}

RecurrentStateGather
====================

**Versioned name**: *RecurrentStateGather*

**Category**: *Internal*

**Short description**: Reset source-cache rows, gather active matrix states, and
reorder a suffix of a shared row map into a full prepared cache.

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

``key_size`` and ``value_size`` are integer attributes, defaulting to zero.
Both must be zero (no transpose) or positive with a product dividing ``F``.
Positive sizes convert each active matrix from cache layout ``[head,value,key]``
to GatedDeltaNet layout ``[head,key,value]``. The full cache keeps its input layout.

**Inputs (zero-based C++ port indices)**

* ``0 cache``: floating-point ``[1,1,S,F]``.
* ``1 active_rows``: integer ``[1,1,1,B]``; each value selects a source row.
* ``2 extra_rows``: integer ``[1,1,1,N]``; only a selected suffix is copied.
* ``3 reset_begin``: integer ``[1,1,1,1]``; first source row to clear.
* ``4 reset_length``: integer ``[1,1,1,1]``; number of source rows to clear.
* ``5 extra_offsets``: integer ``[1,1,1,2]`` containing
  ``(destination_begin, source_begin)``.

All row indices lie in ``[0,S)``. The reset range must fit in ``[0,S]``.
``source_begin`` lies in ``[0,N]``; the selected suffix of length ``N-source_begin``
must fit at ``destination_begin``. A suffix of length zero performs no reorder.

**Outputs**

* ``0 active``: ``[1,1,B,F]`` in cache precision, optionally transposed.
* ``1 prepared_cache``: ``[1,1,S,F]`` in cache precision.

**Reference semantics**

.. code-block:: python

   cleared = cache[0, 0].copy()
   cleared[reset_begin:reset_begin + reset_length] = 0
   active = cleared[active_rows].copy()  # read before any reorder
   prepared = cleared.copy()
   suffix = extra_rows[source_begin:]
   prepared[destination_begin:destination_begin + len(suffix)] = cleared[suffix]
   if key_size > 0:
       active = active.reshape(B, -1, value_size, key_size)
       active = active.transpose(0, 1, 3, 2).reshape(B, F)

For example, map ``[1,0]`` swaps source rows 0 and 1 even if both destination
rows belong to the same cache. It reads both from ``cleared`` rather than
feeding the first write into the second. ``active`` feeds GatedDeltaNet;
``prepared_cache`` feeds :doc:`recurrent-state-update`.
