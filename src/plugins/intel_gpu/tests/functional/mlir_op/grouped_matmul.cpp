// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/grouped_matmul.hpp"

#include <string>
#include <tuple>
#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "mlir_test_env.hpp"

namespace {

using ov::test::GroupedMatMulCompressedLayerTest;
using ov::test::GroupedMatMulLayerTest;
using ov::test::GroupedMatMulShapeParams;
using ov::test::TokensPerExpert;
using ov::test::utils::DecompressionType;

// GroupedMatMul-17, 2Dx3D, f16 -> linalgx.grouped_matmul. Model and inputs come from the
// shared single-op test: A:[T,K] (T dynamic) x B:[E,N,K] (Constant), offsets:[E].
class GroupedMatMulMlirTest : public ov::test::MlirTestFixture<GroupedMatMulLayerTest> {};

// Routing with inactive experts (zero-length groups).
class GroupedMatMulMlirEmptyExpertsTest : public GroupedMatMulMlirTest {};

TEST_P(GroupedMatMulMlirTest, Inference) {
    run();
}

TEST_P(GroupedMatMulMlirEmptyExpertsTest, Inference) {
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

class GroupedMatMulCompressedMlirTest : public ov::test::MlirTestFixture<GroupedMatMulCompressedLayerTest> {
protected:
    // The test graph has several ops (the matmul and its weights decompression), so "at least one MLIROp"
    // from the base fixture isn't enough: also check that the matmul itself didn't stay on cldnn
    void check_mlir_execution() override {
        ov::test::MlirTestFixture<GroupedMatMulCompressedLayerTest>::check_mlir_execution();
        if (!ov::test::is_mlir_enabled()) {
            return;
        }
        // Сheck the matmul didn't stay on cldnn
        ov::test::CheckNumberOfNodesWithType(compiledModel, "grouped_matmul", 0);
    }
};

TEST_P(GroupedMatMulCompressedMlirTest, Inference) {
    const auto& params = GetParam();
    const int group_size = std::get<8>(params);
    if (group_size == -1 || group_size == 32) {
        GTEST_SKIP() << "Unsupported yet: decompression group size " << group_size;
    }
    const auto weights_type = std::get<2>(params);
    const auto zp_type = std::get<6>(params);
    if (zp_type == DecompressionType::empty && weights_type.bitwidth() == 4) {
        GTEST_SKIP() << "Unsupported yet: " << weights_type << " weights without a zero point";
    }
    run();
}

const std::vector<GroupedMatMulShapeParams> shapes_2d_3d_compressed = {
    {{ov::PartialShape{-1, 128}, {{16, 128}, {32, 128}, {8, 128}}}, {4, 256, 128}, TokensPerExpert{{4, 4, 4, 4}, {2, 10, 4, 16}, {1, 3, 2, 2}}},
    {{ov::PartialShape{-1, 256}, {{16, 256}, {32, 256}}}, {8, 512, 256}, TokensPerExpert{{4, 2, 2, 2, 2, 2, 1, 1}, {2, 6, 4, 4, 4, 4, 4, 4}}},
    // Routing with inactive experts (zero-length groups)
    {{ov::PartialShape{-1, 128}, {{40, 128}, {72, 128}}}, {4, 256, 128}, TokensPerExpert{{35, 0, 5, 0}, {20, 0, 40, 12}}},
};

INSTANTIATE_TEST_SUITE_P(mlir_GroupedMatMul_Compressed_2d3d,
                         GroupedMatMulCompressedMlirTest,
                         ::testing::Combine(::testing::ValuesIn(shapes_2d_3d_compressed),
                                            ::testing::Values(ov::element::f16),                                   // activations
                                            ::testing::Values(ov::element::u4, ov::element::i4, ov::element::u8),  // weights
                                            ::testing::Values(ov::element::f16),                                   // decompression
                                            ::testing::Values(ov::element::f16),                                   // scale
                                            ::testing::Values(DecompressionType::full),                            // multiply
                                            ::testing::Values(DecompressionType::full, DecompressionType::empty),  // subtract (zp)
                                            ::testing::Values(true),                                               // reshape on decompression
                                            ::testing::Values(-1, 32, 64),                                         // group size (-1 = per-N)
                                            ::testing::Values(ov::test::utils::DEVICE_GPU),
                                            ::testing::Values(std::string{}),
                                            ::testing::Values(ov::AnyMap{})),
                         GroupedMatMulCompressedLayerTest::getTestCaseName);

}  // namespace
