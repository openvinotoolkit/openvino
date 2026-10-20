.. {#openvino_docs_ops_internal_RecurrentConvCache}

RecurrentConvCache
==================

**Versioned name**: *RecurrentConvCache*

**Category**: *Internal*

**Short description**: Combine reset/gather/reorder, convolution-history
concatenation, and ordered snapshot window writes in one cache operation.

These internal operations were introduced to support Qwen3.5 in the llama.cpp
OpenVINO backend, including the GGML ``qwen35`` and ``qwen35moe`` architectures
used by Qwen3.5 and Qwen3.6 models. They fuse llama.cpp's recurrent-cache data
movement for convolution history (``cache_r``) and GatedDeltaNet matrix state
(``cache_s``). Runtime row maps preserve ``llama-server`` multi-slot reset/reorder
semantics, and token-prefix snapshots support speculative MTP verification and
rollback. The operations do not perform attention or convolution arithmetic.
Native implementations are provided by the Intel GPU plugin; their contracts
are expressed through tensors and runtime metadata rather than model names.

All inputs have rank four. Floating-point inputs support ``f16`` and ``f32``;
metadata inputs support ``i32`` and ``i64`` independently. ``S`` is the cache slot
count, ``F`` the flattened per-slot size, and ``B`` the active sequence count.
Dimensions may be dynamic, but rank must be known. Runtime metadata values must
satisfy the bounds below; the kernel does not validate out-of-range values.

The operation does not modify its input cache. Its GPU kernel reads immutable
source allocations, so arbitrary row-map cycles and repeated sources are safe.
The plugin can stage a caller's shared remote input/output tensor to preserve this
contract; kernel input/output allocations themselves must be distinct.

In the llama.cpp integration, this op maintains ``cache_r``. Output 0 feeds
SSM convolution, and output 1 stores convolution-history windows for subsequent
decode calls or rollback to an accepted MTP token prefix.

**Attributes**

``history`` is a required positive integer ``H``. The cache feature count is
``F=C*H``, with each row laid out as ``[channels,history]``.

**Inputs (zero-based C++ port indices)**

Inputs 0-5 have the shapes and reset/reorder semantics of
:doc:`recurrent-state-gather` (without matrix transpose).

* ``6 qkv``: floating-point ``[1,B,C,T]`` containing new projected tokens.
* ``7 windows``: integer ``[1,1,K,2]``; each pair is
  ``(start_column, destination_row)``. ``K`` may be zero.

Each start lies in ``[0,T]``. Each destination selects ``B`` consecutive cache
rows and must fit within ``[0,S]``. Callers handling padded or short batches
must provide windows corresponding to their valid tokens.

**Outputs**

* ``0 convolution_input``: ``[1,B,C,H+T]`` in QKV precision.
* ``1 updated_cache``: ``[1,1,S,C*H]`` in cache precision.

**Reference semantics**

.. code-block:: python

   # cleared, active, prepared use RecurrentStateGather semantics without transpose.
   active = active.reshape(B, C, H).astype(qkv.dtype)
   convolution_input = concatenate([active, qkv[0]], axis=-1)
   updated = prepared.copy()
   for start, destination in windows:  # later overlapping windows win
       window = convolution_input[:, :, start:start + H]
       updated[destination:destination + B] = window.reshape(B, C * H).astype(cache.dtype)

The reset/reorder is applied before window writes. Inactive rows retain their
prepared-cache values. Mixed-precision window writes pass through convolution
input precision before converting to cache precision.

For ``H=3`` and ``T=2``, ``start=2`` caches the last three columns of the
five-column convolution input. Multiple windows can preserve earlier token-prefix
histories for speculative rollback. The convolution itself consumes output 0
in a separate operation; only output 1 needs to become the externally stored cache.
