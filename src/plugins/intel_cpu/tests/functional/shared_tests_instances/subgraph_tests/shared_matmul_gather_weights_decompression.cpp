// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/shared_matmul_gather_weights_decompression.hpp"

#include "common_test_utils/test_constants.hpp"

using namespace ov::test;

namespace {
TEST_P(SharedMatmulAndGatherWeightsDecompression, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
    check_results();
}

const std::vector<GatherDecompressionShapeParams> input_shapes = {
    {{128, 256}, {{}, {{256, 256}}}, 1, 0},
    {{128, 256}, {{}, {{256, 256}}}, 1, 0, 16},
};
const std::vector<ElementType> weights_precisions = {ov::element::u8, ov::element::u4, ov::element::i4};
const std::vector<ElementType> decompression_precisions = {ov::element::f32};
const std::vector<bool> decompression_subtract = {true, false};

INSTANTIATE_TEST_SUITE_P(smoke_MatmulAndGatherSharedWeightsDecompression,
                         SharedMatmulAndGatherWeightsDecompression,
                         ::testing::Combine(::testing::Values(utils::DEVICE_CPU),
                                            ::testing::ValuesIn(input_shapes),
                                            ::testing::ValuesIn(weights_precisions),
                                            ::testing::ValuesIn(decompression_precisions),
                                            ::testing::ValuesIn(decompression_subtract),
                                            ::testing::Values(true)),
                         SharedMatmulAndGatherWeightsDecompression::getTestCaseName);

}  // namespace
