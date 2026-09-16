// Copyright (C) 2018-2026 Intel Corporation
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
const std::vector<ov::test::utils::DecompressionType> decompression_subtract_types = {
    ov::test::utils::DecompressionType::full,
    ov::test::utils::DecompressionType::empty
};

INSTANTIATE_TEST_SUITE_P(smoke_MatmulAndGatherSharedWeightsDecompression,
                         SharedMatmulAndGatherWeightsDecompression,
                         ::testing::Combine(::testing::Values(utils::DEVICE_CPU),
                                            ::testing::ValuesIn(input_shapes),
                                            ::testing::ValuesIn(weights_precisions),
                                            ::testing::ValuesIn(decompression_precisions),
                                            ::testing::Values(ov::test::utils::DecompressionType::full),
                                            ::testing::ValuesIn(decompression_subtract_types),
                                            ::testing::Values(true)),
                         SharedMatmulAndGatherWeightsDecompression::getTestCaseName);

// F16/bf16 constant followed by a single Convert(f32) shared between Gather and MatMul,
// covering the Convert-Gather fusion for hybrid precisions.
const std::vector<ElementType> hybrid_weights_precisions = {ov::element::bf16, ov::element::f16};

INSTANTIATE_TEST_SUITE_P(smoke_MatmulAndGatherSharedWeightsDecompression_ConvertOnly,
                         SharedMatmulAndGatherWeightsDecompression,
                         ::testing::Combine(::testing::Values(utils::DEVICE_CPU),
                                            ::testing::Values(input_shapes[0]),
                                            ::testing::ValuesIn(hybrid_weights_precisions),
                                            ::testing::ValuesIn(decompression_precisions),
                                            ::testing::Values(ov::test::utils::DecompressionType::empty),
                                            ::testing::Values(ov::test::utils::DecompressionType::empty),
                                            ::testing::Values(true)),
                         SharedMatmulAndGatherWeightsDecompression::getTestCaseName);

}  // namespace
