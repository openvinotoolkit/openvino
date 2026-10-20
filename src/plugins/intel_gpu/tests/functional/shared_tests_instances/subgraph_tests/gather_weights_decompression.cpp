// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/gather_weights_decompression.hpp"

using namespace ov::test;

namespace {

TEST_P(GatherWeightsDecompression, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
    check_results();
}

const std::vector<ov::element::Type> output_precisions = {ov::element::f32, ov::element::f16};
const std::vector<ov::element::Type> weights_precisions = {ov::element::u8,
                                                           ov::element::i8,
                                                           ov::element::u4,
                                                           ov::element::i4};
const std::vector<GatherDecompressionShapeParams> input_shapes_basic = {
    {{2, 5}, {{-1, -1}, {{2, 3}}}, 1, 1},
    {{15, 32}, {{-1, -1}, {{2, 3}}}, 1, 0, 16},
    {{15, 32}, {{-1, -1}, {{2, 3}}}, 0, 0, 16},
    {{2, 5}, {{}, {{2, 3}}}, 1, -1},
    {{15, 16, 2}, {{-1, -1}, {{2, 3}}}, 0, 0},
};
const std::vector<ov::test::utils::DecompressionType> decompression_multiply_types = {
    ov::test::utils::DecompressionType::scalar,
    ov::test::utils::DecompressionType::full,
};
const std::vector<ov::test::utils::DecompressionType> decompression_subtract_types = {
    ov::test::utils::DecompressionType::empty,
    ov::test::utils::DecompressionType::scalar,
    ov::test::utils::DecompressionType::full,
};
const std::vector<bool> reshape_on_decompression = {true, false};

INSTANTIATE_TEST_SUITE_P(smoke_GatherCompressedWeights_basic,
                         GatherWeightsDecompression,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_GPU),
                                            ::testing::ValuesIn(input_shapes_basic),
                                            ::testing::ValuesIn(weights_precisions),
                                            ::testing::ValuesIn(output_precisions),
                                            ::testing::ValuesIn(decompression_multiply_types),
                                            ::testing::ValuesIn(decompression_subtract_types),
                                            ::testing::ValuesIn(reshape_on_decompression)),
                         GatherWeightsDecompression::get_test_case_name);

}  // namespace
