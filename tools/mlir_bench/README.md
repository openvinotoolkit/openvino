# MLP benchmarks

Various MLP benchmarks.
Describes usage of the `*_bench.sh` scripts.

## LIBXSMM
- F32:
```bash
libxsmm_bench.sh
```
- BF16:
```bash
libxsmm_bench.sh -B
```

## Pure MLIR
- F32:
```bash
tpp_mlir_bench.sh -t f32
```
- BF16:
```bash
tpp_mlir_bench.sh -t bf16
```

## OV - no MLIR
Default model:\
`matmul_transpose_b + bias broadcast`

Alternative model - scritp flag `-b mlp`:\
`matmul + bias (no broadcast)`

- F32:
```bash
OV_MLIR=0 mlp_bench.sh -t f32
```
- BF16:
```bash
OV_MLIR=0 mlp_bench.sh -t bf16
```

## OV + MLIR - full
Default model:\
`matmul_transpose_b + bias broadcast`

Alternative model - scritp flag `-b mlp`:\
`matmul + bias (no broadcast)`

- F32:
```bash
OV_MLIR=1 mlp_bench.sh -t f32
```
- BF16:
```bash
OV_MLIR=1 mlp_bench.sh -t bf16
```

## OV + MLIR - kernel only
Default model:\
`matmul_transpose_b + bias broadcast`

Alternative model - scritp flag `-b mlp`:\
`matmul + bias (no broadcast)`

- F32:
```bash
ov_raw_mlir_bench.sh -t f32
```
- BF16:
```bash
ov_raw_mlir_bench.sh -t bf16
```
