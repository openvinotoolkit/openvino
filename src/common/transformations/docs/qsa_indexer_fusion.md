# QSAIndexer Fusion

`ov::pass::FuseQSAIndexer` replaces the supported opset17 `QSAPrepare` and
`SparseAttentionIndexer` path with the common internal `ov::op::internal::QSAIndexer`.
The fused operation returns both selection outputs and the updated block-key and
incomplete-block state.

The pass is intentionally limited to causal QSA with block size 4, a token budget
of at most 512, four query heads, one key head, and head dimension 128. It accepts
the supported internal or decomposed RMSNorm query path with epsilon in
`[1e-6, 1e-5]`. The source indexer must not use optional `index_weights`, and
`QSAPrepare` block-key output must have no consumer other than the matched
indexer. Other precisions are not rejected by the common matcher.

The operation and transformation define graph semantics only. CPU and GPU
execution kernels are separate work and are not provided by this transformation.
