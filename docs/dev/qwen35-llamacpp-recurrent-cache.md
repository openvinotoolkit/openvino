# Qwen3.5 recurrent cache integration in llama.cpp

The default Qwen GPU integration uses dynamic stateless execution with two
internal ops: `RecurrentConvCacheUpdate` and `RecurrentGatedDeltaNetUpdate`.
They support llama.cpp's `qwen35` and `qwen35moe` graphs, including Qwen3.6,
and fuse generic reset, original-row gather, reorder, computation and ordered
writeback for `cache_r` and `cache_s`. Cache row capacity depends on sequence
slots and rollback snapshots, not attention context length. Runtime row maps,
offsets and write controls remain tensor inputs.

## Stable in-place allocations

llama.cpp binds its full cache as both Parameter and Result using one stable USM
allocation. The GPU plugin accepts exact layout/storage matches with exclusive
cache readers, including exclusive optimized Reshape chains. It protects FP32
cache inputs from FP16 precision conversion. Host USM and device USM are both
eligible. There are no rotation banks or pointer swaps.

The first kernel stages required active rows and nonidentity reorder sources in
private device scratch, applying logical reset and computation-precision rounding.
The second kernel computes convolution/GDN and performs ordered cache writes.
The dependency between these launches protects cycles and duplicate sources.
Identity reorder sources need no scratch writes. Preserved rows are still read
for precision compatibility, so eliminating the spare bank does not eliminate
all cache memory traffic. Scratch capacity is conservative.

Shared original-cache readers, incompatible layouts, static models or unrelated
input overlaps reject mutation. These cases retain immutable input semantics,
private output and any required Result copy-back. Distinct caller allocations
execute the compute stage without the gather stage. A failed aliased inference
can partly update caller memory; clear or restore it before retrying. GPU replay
is disabled because stage selection can change with runtime bindings.

## Fused convolution

`RecurrentConvCacheUpdate` takes nine required inputs, including depthwise weights
as input 8. It produces convolution `[1,B,T,C]` and the full updated cache as
output 1 without an intermediate history/QKV tensor. The history-only mode used
to compare ordinary GroupConvolution is removed. Cache reset, reorder and snapshot
precedence are unchanged.

GDN always emits attention as output 0 and the full updated state cache as output
1. It retains the newest `min(T,snapshot_count)` states for MTP, newest first.
The ordinary GDN snapshot support remains available for unmatched graphs.

Specifications: `recurrent-conv-cache-update.rst` and
`recurrent-gated-delta-net-update.rst` in the internal operation documentation.
Native tests in `recurrent_cache_update.cpp` compare convolution against a host
calculation and GDN against the ordinary GDN op with independent host cache
mapping. They cover reset, cycles, duplicates, overlapping writes, mixed
precision, changing shapes, aliased/distinct allocations, exclusive Reshapes,
shared readers, unrelated overlaps, host/device USM and export/import. Core shape
and attribute tests cover the nine-input convolution contract, rejection of
missing weights and GDN snapshots.

## Companion frontend

The llama.cpp frontend uses stable pure-top-k MTP sampling branches with packed
runtime row controls and private inactive outputs. Active slot subsets reuse one
compatible draft model; trunk and other graph configurations retain separate
entries. This is a frontend translation change, not another native cache op.
Unified attention KV is the recommended multi-slot layout because the current
stateless attention KV fusion supports a single physical stream.

Current configuration, validation, graph examples and measured results are indexed
in llama.cpp at `ggml/src/ggml-openvino/dev_doc/README.md`. Historical rotation and
separate-convolution comparisons are evidence for the selected default, not
additional supported implementation modes.
