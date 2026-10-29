# QSAIndexer Fusion

`ov::pass::FuseQSAIndexer` replaces the supported opset17 `QSAPrepare` and
`SparseAttentionIndexer` path with the common internal `ov::op::internal::QSAIndexer`.
The fused operation returns both selection outputs and the updated block-key and
incomplete-block state.

The pass limits are aligned with the GPU `QSAIndexerFusion` (PR #38902). It fuses
only causal QSA with one selection head, the same key block size and key length
in both source ops, a static query head count, and a static head size shared by
the query and the raw keys. RoPE must be enabled with a static rotary dimension,
both RMSNorm weights must be enabled, and the query and key RMSNorm epsilons must
agree within a relative tolerance of `1e-4`. It accepts the internal or decomposed
RMSNorm query path and a `QSAPrepare` whose outputs have no consumer other than the
matched subgraph. The source indexer must not use optional `index_weights`. Head
count, head size, block size and token budget values are not fixed; plugin kernels
reject values they do not support during compilation. Other precisions are not
rejected by the common matcher.

The operation and transformation define graph semantics only. CPU and GPU
execution kernels are separate work and are not provided by this transformation.
