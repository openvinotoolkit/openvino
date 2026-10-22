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
cache ops are included in the rotation commit because the earlier llama.cpp
integration used them to construct and recognize fused graphs. The latest
frontend emits the fused computation/cache ops directly; these foundation
operations remain available for shape validation and standalone reference tests.

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

## Staged in-place Update operations

The second commit adds separate `RecurrentConvCacheUpdate` and
`RecurrentGatedDeltaNetUpdate` op classes for the same llama.cpp Qwen3.5 cache
contract. It builds on the rotation commit's generic state validation,
computation reference tests, precision handling and device-USM output binding.
It changes no existing OpenVINO op contract or computation kernel. Therefore
this is a stacked change: the first commit runs independently; the second
requires the first. Both commits share the same clean upstream master base.

With a compatible exclusive cache reader, llama.cpp may bind one device-USM
allocation as both Parameter and Result. The first GPU kernel gathers required
original active rows and nonidentity reorder sources into device scratch,
applying logical reset and computation-precision rounding. The second kernel
performs convolution/GDN, reset, reorder and ordered writeback in the original
allocation. The gather finishes before any writes, so cyclic and duplicate maps
remain valid. Identity reorder sources need no scratch copy. Retained rows are
still read for rounding and scratch capacity is conservative; removing the
spare bank does not guarantee lower total allocation size.

The plugin permits this mutation only for compatible full-cache layouts, an
exclusive memory-identical view/Reshape chain and no unrelated overlapping
inputs. Rejected alias bindings use private output storage and the required
final copy. Distinct bindings use immutable input semantics with one compute
kernel. A failed aliased inference can leave the original cache partly updated;
the caller must clear or restore it before retrying. Runtime alias changes
also disable command replay for the new implementations.

Enable the experimental llama.cpp path with
`GGML_OPENVINO_RECURRENT_CACHE_INPLACE=1`. Rotation remains the default.
On the earlier target UD measurement with `-np 4 -c 8192`, one active request,
1,024 prompt tokens and 512 generated tokens, in-place improved non-MTP decode
from 27.875 to 29.001 tokens/s, while two-token MTP decreased from 46.345 to
45.014 tokens/s. Those results predate this upstream rebase and are not new
performance claims for these commits. The llama.cpp backend performance report
contains the full method and limitations.

Update specifications are in `recurrent-conv-cache-update.rst` and
`recurrent-gated-delta-net-update.rst` beside the rotation specifications.
`recurrent_cache_update.cpp` native tests cover changing batches/tokens, resets,
cycles, duplicates, snapshots, mixed precision, alternating aliased/distinct
bindings, exclusive Reshapes, unrelated readers/overlaps, and export/import.
