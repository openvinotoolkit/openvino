# QSAIndexer Fusion

`ov::pass::FuseQSAIndexer` replaces the supported opset17 `QSAPrepare` and
`SparseAttentionIndexer` path with the common internal
`ov::op::internal::QSAIndexer`. The fused operation returns both selection outputs
and the updated block-key and incomplete-block state.

The pass matches the internal or decomposed RMSNorm query path, enabled query and
key norm weights, the same key block size and key length in both source ops, and a
`QSAPrepare` whose outputs have no consumer other than the matched subgraph. The
source indexer must not use optional `index_weights`. Input ranks are the
`QSAIndexer` layout constants.

Kernel limits differ between devices, so they are passed to the pass as
`FuseQSAIndexer::Config`: allowed `num_selection_heads` values, the relative
tolerance between the query and key RMSNorm epsilons, whether the head count, head
size and rotary dimension must be static, and whether RoPE is required. The default
configuration fuses every graph `QSAIndexer` can represent. The GPU plugin passes
the limits of the GPU QSAIndexer kernels from openvinotoolkit/openvino#38902 (one
selection head, epsilons within a relative `1e-4`, static head dimensions, RoPE
required); the CPU plugin uses the same limits until it has its own kernel. Head
count, head size, block size and token budget values are not fixed. Other precisions
are not rejected by the common matcher.

The operation and transformation define graph semantics only. CPU and GPU execution
kernels are separate work and are not provided by this transformation.
