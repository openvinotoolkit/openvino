// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/grouped_matmul.hpp"

#include <string>
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

class GroupedMatMulCompressedMlirTest : public ov::test::MlirTestFixture<GroupedMatMulCompressedLayerTest> {
protected:
    // The test graph has several ops (the matmul and its weights decompression), so "at least one MLIROp"
    // from the base fixture isn't enough: also check that the matmul itself didn't stay on cldnn
    void check_mlir_execution() override {
        ov::test::MlirTestFixture<GroupedMatMulCompressedLayerTest>::check_mlir_execution();
        if (!ov::test::is_mlir_enabled()) {
            return;
        }
        // The compressed op must be taken by the MLIR path rather than falling back to cldnn.
        for (const auto& node : compiledModel.get_runtime_model()->get_ordered_ops()) {
            const auto& rt_info = node->get_rt_info();
            const auto it = rt_info.find(ov::exec_model_info::LAYER_TYPE);
            EXPECT_FALSE(it != rt_info.end() && it->second.as<std::string>() == "grouped_matmul")
                << "GroupedMatMul was executed by the cldnn 'grouped_matmul' primitive instead of the MLIR path";
        }
    }
};

TEST_P(GroupedMatMulCompressedMlirTest, Inference) {
    run();
}

const std::vector<GroupedMatMulShapeParams> shapes_2d_3d_compressed = {
    {{ov::PartialShape{-1, 128}, {{16, 128}, {32, 128}, {8, 128}}}, {4, 256, 128}, TokensPerExpert{{4, 4, 4, 4}, {2, 10, 4, 16}, {1, 3, 2, 2}}},
    {{ov::PartialShape{-1, 256}, {{16, 256}, {32, 256}}}, {8, 512, 256}, TokensPerExpert{{4, 2, 2, 2, 2, 2, 1, 1}, {2, 6, 4, 4, 4, 4, 4, 4}}},
};

INSTANTIATE_TEST_SUITE_P(mlir_GroupedMatMul_Compressed_2d3d,
                         GroupedMatMulCompressedMlirTest,
                         ::testing::Combine(::testing::ValuesIn(shapes_2d_3d_compressed),
                                            ::testing::Values(ov::element::f16),                                   // activations
                                            ::testing::Values(ov::element::u4, ov::element::i4),                   // weights
                                            ::testing::Values(ov::element::f16),                                   // decompression
                                            ::testing::Values(ov::element::f16),                                   // scale
                                            ::testing::Values(DecompressionType::full),                            // multiply
                                            ::testing::Values(DecompressionType::full, DecompressionType::empty),  // subtract (zp)
                                            ::testing::Values(true),                                               // reshape on decompression
                                            ::testing::Values(-1, 64),                                             // group size (-1 = per-N)
                                            ::testing::Values(ov::test::utils::DEVICE_GPU),
                                            ::testing::Values(std::string{}),
                                            ::testing::Values(ov::AnyMap{})),
                         GroupedMatMulCompressedLayerTest::getTestCaseName);

// Real MoE-like shape: E=256 experts, N=512, K=2048, u4 weights + u4 zp, group size 64 (G=32), i.e.
//   weights memref<256x512x2048xi4>, scale/zp memref<256x512x32x{f16,i4}, strided<[16384, 1, 512]>>.
// Empty tokens_per_expert -> tokens are spread evenly over the experts.
const std::vector<GroupedMatMulShapeParams> shapes_2d_3d_compressed_moe = {
    {{ov::PartialShape{-1, 2048}, {{256, 2048}, {1024, 2048}}}, {256, 512, 2048}, TokensPerExpert{}},
};

INSTANTIATE_TEST_SUITE_P(mlir_GroupedMatMul_Compressed_2d3d_moe,
                         GroupedMatMulCompressedMlirTest,
                         ::testing::Combine(::testing::ValuesIn(shapes_2d_3d_compressed_moe),
                                            ::testing::Values(ov::element::f16),         // activations
                                            ::testing::Values(ov::element::u4),          // weights
                                            ::testing::Values(ov::element::f16),         // decompression
                                            ::testing::Values(ov::element::f16),         // scale
                                            ::testing::Values(DecompressionType::full),  // multiply
                                            ::testing::Values(DecompressionType::full),  // subtract (zp)
                                            ::testing::Values(true),                     // reshape on decompression
                                            ::testing::Values(64),                       // group size -> G = 2048 / 64 = 32
                                            ::testing::Values(ov::test::utils::DEVICE_GPU),
                                            ::testing::Values(std::string{}),
                                            ::testing::Values(ov::AnyMap{})),
                         GroupedMatMulCompressedLayerTest::getTestCaseName);

}  // namespace
