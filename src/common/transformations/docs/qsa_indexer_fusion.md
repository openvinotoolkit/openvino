# QSAIndexer Fusion

`ov::pass::FuseQSAIndexer` replaces the supported opset17 `QSAPrepare` and
`SparseAttentionIndexer` path with the common internal `ov::op::internal::QSAIndexer`.
The fused operation returns both selection outputs and the updated block-key and
incomplete-block state.

The pass matches the graph structure only: the internal or decomposed RMSNorm
query path, enabled query and key norm weights, a shared key length and block
size, and a `QSAPrepare` whose outputs have no consumer other than the matched
subgraph. The source indexer must not use optional `index_weights`. Head count,
head size, block size, token budget and epsilon values are taken from the graph
and are not restricted by the pass or by `QSAIndexer` shape inference; plugins
reject configurations their kernels do not support during compilation. Other
precisions are not rejected by the common matcher.

The operation and transformation define graph semantics only. CPU and GPU
execution kernels are separate work and are not provided by this transformation.
