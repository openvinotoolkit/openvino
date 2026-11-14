# Qwen3.5 recurrent cache integration in llama.cpp

The default Qwen GPU integration uses dynamic stateless execution with two
internal ops: `RecurrentConvCacheUpdate` and `RecurrentGatedDeltaNetUpdate`.
They support llama.cpp's `qwen35` and `qwen35moe` graphs, including Qwen3.6,
and fuse generic reset, original-row gather, reorder, computation and ordered
writeback for `cache_r` and `cache_s`. Cache row capacity depends on sequence
slots and rollback snapshots, not attention context length. Runtime row maps,
offsets and write controls remain tensor inputs.

## Writable cache inputs

llama.cpp binds each full cache as a writable Parameter using a stable GPU USM
allocation. Both ops update that input in place and return only their computation
result. There are no cache Results, output alias bindings, rotation banks or pointer
swaps. The primitive uses the ordinary output allocation path for its single output;
no changes to primitive_inst or network output binding are needed.

The GPU precision pass protects caller-owned cache Parameters and exclusive
Reshape aliases. A cache with another graph reader or a computed temporary cache
bank is rejected. The caller must bind a writable GPU RemoteTensor and ensure
other inputs and the computation output do not overlap its allocation. Ordinary host inputs are copied to GPU
storage and do not provide caller-visible cache updates.

The first kernel stages active rows and nonidentity reorder sources in private
device scratch, applying logical reset and computation-precision rounding. The
second computes convolution/GDN and performs ordered cache writes. Their dependency
protects cycles and duplicate sources. Identity reorder rows need no scratch copy.
A failed inference can leave caller state partly updated; clear or restore it before
retrying.

## Computation outputs

`RecurrentConvCacheUpdate` takes nine inputs, including depthwise weights, and
returns convolution `[1,B,T,C]`. `RecurrentGatedDeltaNetUpdate` takes twelve inputs
and returns attention `[B,T,H,V]`. GDN stores the newest `min(T,snapshot_count)`
states in its cache input, newest first. Ordinary GDN is unchanged. Unmatched
llama.cpp graphs obtain state snapshots with ordinary GDN token prefixes and
functional writeback instead of mutating a temporary bank.

Native tests compare convolution against a host calculation and GDN against
ordinary GDN prefixes with independent host cache mapping. They cover reset,
cycles, duplicates, overlapping writes, mixed precision, changing shapes, input
rebinding, exclusive Reshapes, shared-reader rejection, host/device USM and
compiled-model export/import. Core shape/attribute tests enforce one output.

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
