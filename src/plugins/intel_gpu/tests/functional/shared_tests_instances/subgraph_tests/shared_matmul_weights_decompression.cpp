// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/shared_matmul_weights_decompression.hpp"

#include "common_test_utils/test_constants.hpp"
#include "openvino/op/subtract.hpp"

using namespace ov::test;

namespace {
void check_runtime_model(const std::shared_ptr<const ov::Model>& runtime_model) {
    const auto& ops = runtime_model->get_ops();
    ASSERT_TRUE(std::find_if(ops.begin(), ops.end(), [](const std::shared_ptr<const ov::Node>& node) {
        return node->get_type_info() == ov::op::v1::Multiply::get_type_info_static() ||
               node->get_type_info() == ov::op::v1::Subtract::get_type_info_static() ||
               node->get_type_info() == ov::op::v0::Convert::get_type_info_static();
    }) == ops.end());
}

TEST_P(SharedMatmulWeightsDecompression, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
    check_runtime_model(compiledModel.get_runtime_model());
}
const std::vector<MatMulDecompressionShapeParams> input_shapes = {
    {{{}, {{1, 1, 256}}}, {256, 128}},
    {{{}, {{1, 1, 256}}}, {256, 128}, 64ul},
};
const std::vector<ElementType> decompression_precisions = {ov::element::f16, ov::element::f32};
const std::vector<ElementType> weights_precisions = {ov::element::u8, ov::element::u4};
const std::vector<bool> transpose_weights = {true, false};

INSTANTIATE_TEST_SUITE_P(smoke_MatMulSharedCompressedWeights,
                         SharedMatmulWeightsDecompression,
                         ::testing::Combine(::testing::Values(utils::DEVICE_GPU),
                                            ::testing::ValuesIn(input_shapes),
                                            ::testing::ValuesIn(weights_precisions),
                                            ::testing::ValuesIn(decompression_precisions),
                                            ::testing::ValuesIn(transpose_weights),
                                            ::testing::Values(DecompressionType::full),
                                            ::testing::Values(true)),
                         SharedMatmulWeightsDecompression::getTestCaseName);

}  // namespace
