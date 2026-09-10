// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/test_constants.hpp"
#include "openvino/runtime/properties.hpp"
#include "subgraph_tests/conv1x1_weight_compressed_to_matmul.hpp"

namespace {
using ov::test::Conv1x1ExpectedOpCounts;
using ov::test::Conv1x1WeightCompressedShapeParams;
using ov::test::Conv1x1WeightCompressedToMatmulTest;
using ov::test::InputShape;

const std::vector<ov::element::Type> weights_precisions = {ov::element::i4, ov::element::i8};

// On ARM the compressed-weights path is only numerically correct through KleidiAI, which requires
// disabling dynamic activation quantization (same convention as smoke_MatMulCompressedWeights_Kleidiai
// and smoke_GroupedMatMul_Compressed). Platforms without the required ISA are skipped in
// skip_tests_config.cpp, mirroring how the Kleidiai suite is gated.
#if defined(OPENVINO_ARCH_ARM64) || defined(OPENVINO_ARCH_ARM)
const ov::AnyMap config = {ov::hint::dynamic_quantization_group_size(UINT64_MAX)};
#else
const ov::AnyMap config = {};
#endif

const Conv1x1ExpectedOpCounts op_counts_1x1{{"Convolution", 0}, {"FullyConnected", 1}, {"Transpose", 0}, {"Reshape", 1}};

const InputShape transpose_in_1x1{{-1, 1, 1, 128}, {{1, 1, 1, 128}, {3, 1, 1, 128}}};
// input Transpose, H*W == 1: the output decoration does not change the (fully eliminated) layout ops.
INSTANTIATE_TEST_SUITE_P(
    smoke_Conv1x1WeightCompressedToMatmul_TransposeInput,
    Conv1x1WeightCompressedToMatmulTest,
    ::testing::Combine(::testing::Values(Conv1x1WeightCompressedShapeParams{transpose_in_1x1, 128}),
                       ::testing::Values("Transpose"),
                       ::testing::Values("Transpose", "Reshape"),
                       ::testing::Values(ov::element::f32),
                       ::testing::ValuesIn(weights_precisions),
                       ::testing::Values(op_counts_1x1),
                       ::testing::Values(config),
                       ::testing::Values(ov::test::utils::DEVICE_CPU)),
    Conv1x1WeightCompressedToMatmulTest::getTestCaseName);

const InputShape reshape_in_1x1{{-1, 128}, {{1, 128}, {3, 128}}};

INSTANTIATE_TEST_SUITE_P(
    smoke_Conv1x1WeightCompressedToMatmul_ReshapeInput,
    Conv1x1WeightCompressedToMatmulTest,
    ::testing::Combine(::testing::Values(Conv1x1WeightCompressedShapeParams{reshape_in_1x1, 128}),
                       ::testing::Values("Reshape"),
                       ::testing::Values("Reshape"),
                       ::testing::Values(ov::element::f32),
                       ::testing::ValuesIn(weights_precisions),
                       ::testing::Values(op_counts_1x1),
                       ::testing::Values(config),
                       ::testing::Values(ov::test::utils::DEVICE_CPU)),
    Conv1x1WeightCompressedToMatmulTest::getTestCaseName);

// H*W > 1: a 1x1 Convolution producing [N, 128, 1, 55] consumed by a Reshape. Transpose after MatMul is needed
const InputShape transpose_in_1x55{{-1, 1, 55, 128}, {{1, 1, 55, 128}, {3, 1, 55, 128}}};
const Conv1x1ExpectedOpCounts op_counts_1xW{{"Convolution", 0}, {"FullyConnected", 1}, {"Transpose", 1}, {"Reshape", 1}};

INSTANTIATE_TEST_SUITE_P(
    smoke_Conv1x1WeightCompressedToMatmul_SpatialGt1,
    Conv1x1WeightCompressedToMatmulTest,
    ::testing::Combine(::testing::Values(Conv1x1WeightCompressedShapeParams{transpose_in_1x55, 128}),
                       ::testing::Values("Transpose"),
                       ::testing::Values("Reshape"),
                       ::testing::Values(ov::element::f32),
                       ::testing::ValuesIn(weights_precisions),
                       ::testing::Values(op_counts_1xW),
                       ::testing::Values(config),
                       ::testing::Values(ov::test::utils::DEVICE_CPU)),
    Conv1x1WeightCompressedToMatmulTest::getTestCaseName);

}  // namespace
