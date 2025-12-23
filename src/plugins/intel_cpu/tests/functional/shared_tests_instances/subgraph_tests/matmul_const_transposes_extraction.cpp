// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/matmul_const_transposes_extraction.hpp"

using namespace ov::test;

namespace {
std::vector<MatMulConstTransposesExtractionTestShapeParams> shape_params = {
    {{2, 2}, {2, 3}, false},
    {{5}, {5, 1}, false},
    {{5}, {5, 3}, false},
    {{5, 10}, {10, 7}, false},
    {{5, 10}, {1, 10, 7}, false},
    {{5, 10}, {1, 1, 10, 7}, false},
    {{2, 3, 5, 10}, {10, 7}, false},
    {{2, 3, 5, 10}, {1, 10, 7}, false},
    {{2, 3, 5, 10}, {1, 10, 1}, false},
    {{2, 3, 5, 10}, {1, 1, 10, 7}, false},
    {{2, 3, 5, 10}, {1, 1, 10, 1}, false},
};

INSTANTIATE_TEST_SUITE_P(smoke_MatMulConstTransposesExtractionTest,
                         MatMulConstTransposesExtractionTest,
                         ::testing::Combine(::testing::ValuesIn(shape_params),
                                            ::testing::Values(true),  // can be fused
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         MatMulConstTransposesExtractionTest::getTestCaseName);

std::vector<MatMulConstTransposesExtractionTestShapeParams> negative_shape_params = {
    {{5}, {5}, false},
    {{5}, {3, 5}, true},
    {{5, 5}, {5, 5}, true},
    {{5, 10}, {7, 10}, true},
    {{5, 10}, {2, 10, 7}, false},
    {{5, 10}, {2, 3, 10, 7}, false},
    {{1, 1, 5, 10}, {10}, false},
    {{1, 1, 5, 10}, {7, 10}, true},
    {{1, 1, 5, 10}, {1, 1, 7, 10}, true},
    {{2, 3, 5, 10}, {7, 10}, true},
    {{2, 3, 5, 10}, {3, 7, 10}, true},
    {{2, 3, 5, 10}, {2, 3, 7, 10}, true},
    {{2, 3, 5, 10}, {3, 10, 7}, false},
    {{2, 3, 5, 10}, {1, 3, 10, 7}, false},
    {{2, 3, 5, 10}, {2, 3, 10, 7}, false},
};

INSTANTIATE_TEST_SUITE_P(smoke_NegativeMatMulConstTransposesExtractionTest,
                         MatMulConstTransposesExtractionTest,
                         ::testing::Combine(::testing::ValuesIn(negative_shape_params),
                                            ::testing::Values(false),  // cannot be fused
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         MatMulConstTransposesExtractionTest::getTestCaseName);

std::vector<MatMulConstTransposesExtractionTestShapeParams> shape_params2 = {
    {{2, 2}, {2, 2}, false},
    {{5, 10}, {10, 7}, false},
    {{5, 10}, {1, 10, 7}, false},
    {{5, 10}, {1, 1, 10, 7}, false},
    {{2, 3, 5, 10}, {10, 7}, false},
    {{2, 3, 5, 10}, {1, 10, 7}, false},
    {{2, 3, 5, 10}, {1, 1, 10, 7}, false},
};

INSTANTIATE_TEST_SUITE_P(smoke_QuantizedMatMulConstTransposesExtractionTest,
                         QuantizedMatMulConstTransposesExtractionTest,
                         ::testing::Combine(::testing::ValuesIn(shape_params2),
                                            ::testing::Values(true),  // can be fused
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         QuantizedMatMulConstTransposesExtractionTest::getTestCaseName);

}  // namespace
