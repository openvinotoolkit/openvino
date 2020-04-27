#include "variadic_split_tests.hpp"

TEST_P(VariadicSplitTests, smoke_GPU_TestsVariadicSplit) {}

INSTANTIATE_TEST_CASE_P(
    smoke_TestsVariadicSplit, VariadicSplitTests,
    ::testing::Values(
        variadic_split_params{ "GPU", 1, {2, 4}, {1, 6, 22, 22}, {{1, 2, 22, 22}, {1, 4, 22, 22}} },
        variadic_split_params{ "GPU", 1, {4, 6}, {1, 10, 22, 22}, {{1, 4, 22, 22}, {1, 6, 22, 22}} },
        variadic_split_params{ "GPU", 1, {2, 4, 1}, {1, 7, 22, 22}, {{1, 2, 22, 22}, {1, 4, 22, 22}, {1, 1, 22, 22}} }, 
        variadic_split_params{ "GPU", 2, {10, 6}, {1, 10, 16, 22}, {{1, 10, 10, 22}, {1, 10, 6, 22}} },
        variadic_split_params{ "GPU", 3, {2, 4, 9, 10, 11}, {1, 5, 5, 36}, {{1, 5, 5, 2}, {1, 5, 5, 4}, {1, 5, 5, 9}, {1, 5, 5, 10}, {1, 5, 5, 11}} }
));
