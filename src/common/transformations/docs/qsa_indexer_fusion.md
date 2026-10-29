# QSAIndexer Fusion

`ov::pass::FuseQSAIndexer` replaces the supported opset17 `QSAPrepare` and
`SparseAttentionIndexer` path with the common internal `ov::op::internal::QSAIndexer`.
The fused operation returns both selection outputs and the updated block-key and
incomplete-block state.

The pass matches the graph structure only: the internal or decomposed RMSNorm
query path, enabled query and key norm weights, a shared key length, and a
`QSAPrepare` whose outputs have no consumer other than the matched subgraph. The
source indexer must not use optional `index_weights`. Kernel limits (causal QSA
with block size 4, a token budget of at most 512, four query heads, head
dimension 128, and epsilon in `[1e-6, 1e-5]`) are exposed by
`QSAIndexer::is_supported` and are checked by plugins during compilation, not by
the pass. Other precisions are not rejected by the common matcher.

The operation and transformation define graph semantics only. CPU and GPU
execution kernels are separate work and are not provided by this transformation.
