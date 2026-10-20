# Test Debug Capabilities

Environment variables that influence test diagnostics output.

## Tensor comparison

Used by `ov::test::utils::compare()` (see `common_test_utils/src/ov_tensor_utils.cpp`).

### `OV_TEST_MAX_DIFFS_TO_PRINT`

Maximum number of mismatched elements to print on comparison failure.

- Default: `0` — only the coordinate with the largest diff is printed.
- Any positive integer — print up to N first mismatched coordinates in addition to the max diff line.
- `-1` — print all mismatched coordinates.
- Invalid / non-numeric values yield `0`.

Example:

```sh
OV_TEST_MAX_DIFFS_TO_PRINT=32 ./bin/intel64/Release/ov_cpu_func_tests --gtest_filter=...
```
