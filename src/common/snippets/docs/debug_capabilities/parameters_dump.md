# Snippet parameters dump

The pass dumps selected properties of some performance-critical operations in Subgraphs. Only MatMuls are currently supported by this pass.

To turn on snippet properties dump feature, the following environment variable should be used:
```sh
    OV_SNIPPETS_DUMP_BRGEMM_PARAMS="path=<path_to_csv_dump_file>" binary ...
```

Examples:
```sh
    OV_SNIPPETS_DUMP_BRGEMM_PARAMS="path=brgemm.csv" binary ...
```

Output example:

| subgraph_name      | name       | in_type     | out_type | in_shapes                           | out_shapes           | in_layouts               | out_layouts | M   | N   | K   | m_block | n_block  | k_block  | acc_max_time  | avg_max_time  |
|--------------------|------------|-------------|----------|-------------------------------------|----------------------|--------------------------|-------------|-----|-----|-----|---------|----------|----------|---------------|---------------|
| FakeQuantitze_457  | MatMul_438 | i8;i8;f32   | i32      | 1 16 128 64;1 16 64 128;1 16 64 128 | 1 16 128 128         | 0 2 1 3;0 1 2 3;0 1 2 3; | 0 1 2 3;    | 128 | 128 | 64  | 32      | FULL_DIM | FULL_DIM | 41482         | 5185          |
| FakeQuantitze_457  | MatMul_452 | u8;i8       | i32      | 1 16 128 128;1 16 128 64            | 1 16 128 64          | 0 1 2 3;0 1 2 3;         | 0 1 2 3;    | 128 | 64  | 128 | 32      | FULL_DIM | FULL_DIM | 39427         | 4928          |
