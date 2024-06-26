// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/matmul_split_decompose.hpp"

using namespace ov::test;
namespace {

std::vector<MatMulGatherDecomposeShapeParams> mm_gather_shape_params = {
    // {{B, L, H*S}, {3*H*S, H*S}, true, true, {1, 1, 3*H*S}, {B, L, 3/*QKV*/, H, S}},
    // {{1, 197, 768}, {2304, 768}, true, true, {1, 1, 2304}, {1, 197, 3, 12, 64}},
    {{2, 5, 8}, {24, 8}, true, true, {1, 1, 24}, {2, 5, 3, 2, 4}},
    {{1, 1, 8}, {24, 8}, true, false, {1, 1, 24}, {1, 1, 3, 2, 4}},
};

INSTANTIATE_TEST_SUITE_P(smoke_MatMulGatherDecompose,
                         MatMulGatherDecompose,
                         ::testing::Combine(::testing::ValuesIn(mm_gather_shape_params),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU),
                                            ::testing::Values(false, true)),
                         MatMulGatherDecompose::getTestCaseName);

std::vector<MatMulSplitDecomposeShapeParams> mm_split_shape_params = {
    // {{1, 1, 5120}, {7680, 5120}, true, {5120, 1280, 1280}, {1, 1, 40, 128}, {1, 1, 10, 128}, {1, 1, 10, 128}},
    {{1, 1, 32}, {128, 32}, true, {64, 32, 32}, {1, 1, 4, 16}, {1, 1, 1, 32}, {1, 1, 1, 32}},
    {{2, 2, 32}, {32, 128}, false, {64, 32, 32}, {2, 2, 4, 16}, {2, 2, 1, 32}, {2, 2, 1, 32}},
    {{1, 16}, {36, 16}, true, {12, 12, 12}, {1, 2, 6}, {1, 1, 12}, {1, 1, 12}},
};

INSTANTIATE_TEST_SUITE_P(smoke_MatMulSplitDecompose,
                         MatMulSplitDecompose,
                         ::testing::Combine(::testing::ValuesIn(mm_split_shape_params),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         MatMulSplitDecompose::getTestCaseName);

}  // namespace
