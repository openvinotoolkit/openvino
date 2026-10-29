# QSAIndexer Fusion

`ov::pass::FuseQSAIndexer` replaces the supported opset17 `QSAPrepare` and
`SparseAttentionIndexer` path with the common internal `ov::op::internal::QSAIndexer`.
The fused operation returns both selection outputs and the updated block-key and
incomplete-block state.

The pass fuses only graphs within the QSAIndexer kernel limits, which are defined
once as `QSAIndexer` constants: causal QSA with block size 4, one selection head,
a positive token budget of at most 512 divisible by the block size, four query
heads, head dimension 128, and query and key RMSNorm epsilon in `[1e-6, 1e-5]`.
It accepts the internal or decomposed RMSNorm query path with enabled query and
key norm weights, a shared key length, and a `QSAPrepare` whose outputs have no
consumer other than the matched subgraph. The source indexer must not use
optional `index_weights`. `QSAIndexer` shape inference itself does not enforce
these limits. Other precisions are not rejected by the common matcher.

The operation and transformation define graph semantics only. CPU and GPU
execution kernels are separate work and are not provided by this transformation.
