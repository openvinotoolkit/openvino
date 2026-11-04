// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/grouped_matmul.hpp"

#include <string>
#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "mlir_test_env.hpp"

namespace {

using ov::test::GroupedMatMulLayerTest;
using ov::test::GroupedMatMulShapeParams;
using ov::test::TokensPerExpert;

// GroupedMatMul-17, 2Dx3D, f16 -> linalgx.grouped_matmul. Model and inputs come from the
// shared single-op test: A:[T,K] (T dynamic) x B:[E,N,K] (Constant), offsets:[E].
class GroupedMatMulMlirTest : public ov::test::MlirTestFixture<GroupedMatMulLayerTest> {};

// Routing with inactive experts (zero-length groups).
class GroupedMatMulMlirEmptyExpertsTest : public GroupedMatMulMlirTest {};

TEST_P(GroupedMatMulMlirTest, Inference) {
    run();
}

TEST_P(GroupedMatMulMlirEmptyExpertsTest, Inference) {
    GTEST_SKIP() << "Known failure: experts without tokens (zero-length groups) are not handled correctly yet";
    run();
}

const std::vector<GroupedMatMulShapeParams> shapes_2d_3d = {
    {{ov::PartialShape{-1, 128}, {{16, 128}, {32, 128}, {8, 128}}}, {4, 256, 128}, TokensPerExpert{{4, 4, 4, 4}, {2, 10, 4, 16}, {1, 3, 2, 2}}},
    {{ov::PartialShape{-1, 256}, {{16, 256}, {32, 256}}}, {8, 512, 256}, TokensPerExpert{{4, 2, 2, 2, 2, 2, 1, 1}, {2, 6, 4, 4, 4, 4, 4, 4}}},
};

const std::vector<GroupedMatMulShapeParams> shapes_2d_3d_empty_experts = {
    {{ov::PartialShape{-1, 128}, {{16, 128}, {32, 128}, {8, 128}}}, {4, 256, 128}, TokensPerExpert{{8, 0, 8, 0}, {0, 16, 0, 16}, {4, 4, 0, 0}}},
};

INSTANTIATE_TEST_SUITE_P(mlir_GroupedMatMul_f16_2d3d,
                         GroupedMatMulMlirTest,
                         ::testing::Combine(::testing::ValuesIn(shapes_2d_3d),
                                            ::testing::Values(ov::element::f16),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU),
                                            ::testing::Values(std::string{})),
                         GroupedMatMulLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(mlir_GroupedMatMul_f16_2d3d_empty_experts,
                         GroupedMatMulMlirEmptyExpertsTest,
                         ::testing::Combine(::testing::ValuesIn(shapes_2d_3d_empty_experts),
                                            ::testing::Values(ov::element::f16),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU),
                                            ::testing::Values(std::string{})),
                         GroupedMatMulLayerTest::getTestCaseName);

}  // namespace
