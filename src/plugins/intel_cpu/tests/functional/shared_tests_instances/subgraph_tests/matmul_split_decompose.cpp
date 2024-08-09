// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/matmul_split_decompose.hpp"

using namespace ov::test;
namespace {

std::vector<MatMulGatherDecomposeShapeParams> mm_gather_shape_params = {
    {{2, 5, 8}, {24, 8}, true, true, {1, 1, 24}, {2, 5, 3, 2, 4}},
    {{1, 1, 8}, {24, 8}, true, false, {1, 1, 24}, {1, 1, 3, 2, 4}},
    {{1, 2, 4}, {4, 12}, false, true, {1, 1, 12}, {1, 2, 3, 2, 2}},
};

INSTANTIATE_TEST_SUITE_P(smoke_MatMulGatherDecompose,
                         MatMulGatherDecompose,
                         ::testing::Combine(::testing::ValuesIn(mm_gather_shape_params),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU),
                                            ::testing::Values(false, true)),
                         MatMulGatherDecompose::getTestCaseName);
}  // namespace
