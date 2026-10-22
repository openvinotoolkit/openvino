# Qwen3.5 recurrent cache integration in llama.cpp

These internal operations support the `qwen35` and `qwen35moe` graph builders
in llama.cpp, including Qwen3.6 GGUFs that use the same architecture. Each
recurrent layer stores convolution history in `cache_r` and Gated Delta Net
(GDN) state in `cache_s`. Those caches contain a fixed number of state rows;
their capacity depends on sequence slots and rollback snapshots rather than
attention context length.

## Immutable inputs and rotating cache banks

`RecurrentConvCache` with nine inputs combines logical cache reset, original-row
gather, depthwise convolution, reorder, and ordered convolution-history writes.
`RecurrentGatedDeltaNet` combines the corresponding GDN computation and
final-state or MTP snapshot writes. Each produces computation output 0 and
the complete updated cache as output 1. Runtime maps, offsets, and write
controls remain tensor inputs, so compiled graphs support changing active
slots, resets, cyclic or duplicate row maps, and rollback.

llama.cpp supplies two device-USM allocations for each cache: the current bank
as Parameter and a spare bank as Result. The GPU kernel reads the current bank
and writes the updated bank directly. On successful inference, llama.cpp swaps
which allocation is current. There is no final full-cache copy to the original
bank. Checkpoint reads and writes resolve the current allocation through the
backend buffer owner. A failed call does not publish the spare bank as current.

The GPU plugin binds cache output 1 directly to a compatible caller-owned
device-USM Result and rejects overlapping input allocations. Private fallback
cache outputs also remain on device. The precision pass preserves cache
storage precision while allowing the established computation precision;
retained cache rows preserve the materialized path's rounding behavior.

`RecurrentStateGather` and `RecurrentStateUpdate` provide the separate generic
cache-maintenance paths used before the frontend's final computation fusions.
The ordinary six-input `GatedDeltaNet` snapshot foundation supports the
unfused MTP path: output 1 is the final state for `snapshot_count=1`, or
newest-first snapshots for a larger count. This foundation and the generic
cache ops are included in the rotation commit because the current llama.cpp
frontend uses them to construct and recognize its fused graphs.

See the operation specifications for exact shapes, rounding and write
precedence:

- `docs/articles_en/documentation/openvino-ir-format/operation-sets/operation-specs/internal/recurrent-conv-cache.rst`
- `docs/articles_en/documentation/openvino-ir-format/operation-sets/operation-specs/internal/recurrent-gated-delta-net.rst`
- `docs/articles_en/documentation/openvino-ir-format/operation-sets/operation-specs/internal/recurrent-state-gather.rst`
- `docs/articles_en/documentation/openvino-ir-format/operation-sets/operation-specs/internal/recurrent-state-update.rst`

The llama.cpp backend selects this dynamic GPU rotation path by default.
Static, CPU and stateful paths retain their separate handling. The native tests
in `src/plugins/intel_gpu/tests/functional/single_layer_tests/recurrent_cache.cpp`
compare against decomposed reference graphs and cover reset, reorder,
overlapping window/snapshot writes, mixed precision, changing shapes, remote
buffers, and export/import. Core shape and attribute tests accompany them.
