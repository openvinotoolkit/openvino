// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/grouped_matmul.hpp"

#include "common_test_utils/subgraph_builders/weights_decompression_builders.hpp"
#include "common_test_utils/test_constants.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/runtime/core.hpp"
#include "transformations/rt_info/decompression.hpp"

namespace {
using namespace ov::test;
using ov::test::utils::DecompressionType;

// Graphs coming from the IR and GGUF readers carry no decompression mark on the f8e8m0 scale Convert,
// while the test builders add one. Clear it so the model matches what those readers produce.
void unmark_e8m0_scale_converts(const std::shared_ptr<ov::Node>& node) {
    if (ov::is_type<ov::op::v0::Convert>(node) && node->get_input_element_type(0) == ov::element::f8e8m0) {
        ov::unmark_as_decompression(node);
    }
    for (const auto& input : node->input_values()) {
        unmark_e8m0_scale_converts(input.get_node_shared_ptr());
    }
}

// Number of tensors of the compiled model held as f8e8m0 (scales that were not widened).
size_t count_e8m0_tensors(const std::shared_ptr<const ov::Model>& runtime_model) {
    size_t count = 0;
    for (const auto& op : runtime_model->get_ordered_ops()) {
        for (const auto& output : op->outputs()) {
            count += output.get_element_type() == ov::element::f8e8m0;
        }
        for (const auto& input : op->inputs()) {
            count += input.get_element_type() == ov::element::f8e8m0;
        }
    }
    return count;
}

const std::vector<ov::element::Type> weights_precisions = {ov::element::u8, ov::element::i8,
                                                           ov::element::u4, ov::element::i4};
const std::vector<ov::element::Type> decompression_precisions = {ov::element::f32};

const std::vector<DecompressionType> sub_decompression_types = {DecompressionType::full,
                                                                DecompressionType::scalar,
                                                                DecompressionType::empty};

const std::vector<GroupedMatMulShapeParams> shapes = {
    // 3D x 3D: A:[G,M,K] x B:[G,N,K] -> [G,M,N], dynamic M dim.
    {{ov::PartialShape{4, -1, 128}, {{4, 8, 128}, {4, 1, 128}, {4, 16, 128}}}, {4, 256, 128}, {}},
    {{ov::PartialShape{8, -1, 256}, {{8, 4, 256}, {8, 1, 256}}}, {8, 512, 256}, {}},
    // 2D x 3D: A:[T,K] x B:[G,N,K] -> [T,N], dynamic T dim.
    {{ov::PartialShape{-1, 128}, {{16, 128}, {32, 128}, {8, 128}}}, {4, 256, 128}, TokensPerExpert{{8, 0, 8, 0}, {0, 16, 0, 16}, {4, 4, 0, 0}}},
    {{ov::PartialShape{-1, 128}, {{12, 128}, {20, 128}}}, {4, 256, 128}, TokensPerExpert{{4, 5, 3, 0}, {0, 7, 6, 7}}},
    {{ov::PartialShape{-1, 256}, {{16, 256}, {32, 256}}}, {8, 512, 256}, TokensPerExpert{{4, 2, 2, 2, 2, 2, 1, 1}, {2, 6, 4, 4, 4, 4, 4, 4}}},
};

INSTANTIATE_TEST_SUITE_P(smoke_GroupedMatMul,
                         GroupedMatMulLayerTest,
                         ::testing::Combine(::testing::ValuesIn(shapes),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU),
                                            ::testing::Values("GatherMatmul")),
                         GroupedMatMulLayerTest::getTestCaseName);

// Corner cases: odd K, non-power-of-2 N
const std::vector<GroupedMatMulShapeParams> shapes_corner_cases = {
    {{ov::PartialShape{4, -1, 77}, {{4, 8, 77}, {4, 1, 77}, {4, 8, 77}}}, {4, 123, 77}, {}},
    {{ov::PartialShape{-1, 77}, {{12, 77}, {20, 77}}}, {4, 123, 77}, TokensPerExpert{{4, 5, 3, 0}, {0, 7, 6, 7}}},
};

INSTANTIATE_TEST_SUITE_P(smoke_GroupedMatMul_CornerCases,
                         GroupedMatMulLayerTest,
                         ::testing::Combine(::testing::ValuesIn(shapes_corner_cases),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU),
                                            ::testing::Values("GatherMatmul")),
                         GroupedMatMulLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GroupedMatMul_Compressed,
                         GroupedMatMulCompressedLayerTest,
                         ::testing::Combine(::testing::ValuesIn(shapes),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::ValuesIn(weights_precisions),
                                            ::testing::ValuesIn(decompression_precisions),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(DecompressionType::full),
                                            ::testing::ValuesIn(sub_decompression_types),
                                            ::testing::Values(false),
                                            ::testing::Values(-1, 16),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU),
                                            ::testing::Values("GatherMatmulCompressed"),
                                            ::testing::Values(ov::AnyMap{})),
                         GroupedMatMulCompressedLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GroupedMatMul_Compressed_CornerCases,
                         GroupedMatMulCompressedLayerTest,
                         ::testing::Combine(::testing::ValuesIn(shapes_corner_cases),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::ValuesIn(weights_precisions),
                                            ::testing::ValuesIn(decompression_precisions),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(DecompressionType::full),
                                            ::testing::ValuesIn(sub_decompression_types),
                                            ::testing::Values(false),
                                            ::testing::Values(-1),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU),
                                            ::testing::Values("GatherMatmulCompressed"),
                                            ::testing::Values(ov::AnyMap{})),
                         GroupedMatMulCompressedLayerTest::getTestCaseName);

// MXFP4 (OCP MX): f4e2m1 weights with one f8e8m0 scale per 32 weights, no zero point.
INSTANTIATE_TEST_SUITE_P(smoke_GroupedMatMul_Compressed_mxfp4,
                         GroupedMatMulCompressedLayerTest,
                         ::testing::Combine(::testing::ValuesIn(shapes),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(ov::element::f4e2m1),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(ov::element::f8e8m0),
                                            ::testing::Values(DecompressionType::full),
                                            ::testing::Values(DecompressionType::empty),
                                            ::testing::Values(false),
                                            ::testing::Values(32),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU),
                                            ::testing::Values("GatherMatmulCompressed"),
                                            ::testing::Values(ov::AnyMap{})),
                         GroupedMatMulCompressedLayerTest::getTestCaseName);

// Expert MXFP4 weights: the f8e8m0 scales must reach the compiled GatherMatmulCompressed node unwidened.
class GroupedMatMulMXFP4ScalesTest : public GroupedMatMulCompressedLayerTest {
protected:
    std::shared_ptr<ov::Node> build_weights() override {
        auto weights = GroupedMatMulCompressedLayerTest::build_weights();
        if (clear_decompression_mark()) {
            unmark_e8m0_scale_converts(weights);
        }
        return weights;
    }

    virtual bool clear_decompression_mark() const {
        return true;
    }

    void validate() override {
        GroupedMatMulCompressedLayerTest::validate();
        EXPECT_GT(count_e8m0_tensors(compiledModel.get_runtime_model()), 0u)
            << "MXFP4 expert scales were widened instead of staying f8e8m0";
    }
};

// Same, with the decompression mark on the scale Convert kept (as the shared builders and FuseMOEExperts produce it).
class GroupedMatMulMXFP4MarkedScalesTest : public GroupedMatMulMXFP4ScalesTest {
    bool clear_decompression_mark() const override {
        return false;
    }
};

TEST_P(GroupedMatMulMXFP4ScalesTest, Inference) {
    run();
}

TEST_P(GroupedMatMulMXFP4MarkedScalesTest, Inference) {
    run();
}

INSTANTIATE_TEST_SUITE_P(smoke_GroupedMatMul_mxfp4_scales_e8m0,
                         GroupedMatMulMXFP4ScalesTest,
                         ::testing::Combine(::testing::ValuesIn(shapes),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(ov::element::f4e2m1),
                                            ::testing::Values(ov::element::f16),
                                            ::testing::Values(ov::element::f8e8m0),
                                            ::testing::Values(DecompressionType::full),
                                            ::testing::Values(DecompressionType::empty),
                                            ::testing::Values(false),
                                            ::testing::Values(32),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU),
                                            ::testing::Values("GatherMatmulCompressed"),
                                            ::testing::Values(ov::AnyMap{})),
                         GroupedMatMulMXFP4ScalesTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GroupedMatMul_mxfp4_marked_scales_e8m0,
                         GroupedMatMulMXFP4MarkedScalesTest,
                         ::testing::Combine(::testing::ValuesIn(shapes),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(ov::element::f4e2m1),
                                            ::testing::Values(ov::element::f16),
                                            ::testing::Values(ov::element::f8e8m0),
                                            ::testing::Values(DecompressionType::full),
                                            ::testing::Values(DecompressionType::empty),
                                            ::testing::Values(false),
                                            ::testing::Values(32),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU),
                                            ::testing::Values("GatherMatmulCompressed"),
                                            ::testing::Values(ov::AnyMap{})),
                         GroupedMatMulMXFP4MarkedScalesTest::getTestCaseName);

// FullyConnected (2D weights) is unchanged: its scales are still widened, no f8e8m0 tensor in the compiled model.
TEST(smoke_FullyConnected_mxfp4_scales, WidenedForFullyConnected) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{4, 128});
    auto weights = ov::test::utils::initMatMulDecompressionSubgraphQuantization(ov::Shape{128, 64},  // [K, N]
                                                                                32,
                                                                                ov::element::f32,
                                                                                ov::element::f4e2m1,
                                                                                ov::element::f16,
                                                                                ov::element::f8e8m0,
                                                                                false,
                                                                                DecompressionType::full,
                                                                                DecompressionType::empty,
                                                                                false);
    unmark_e8m0_scale_converts(weights);
    auto matmul = std::make_shared<ov::op::v0::MatMul>(param, weights);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{matmul}, ov::ParameterVector{param});
    auto compiled = ov::Core().compile_model(model, ov::test::utils::DEVICE_CPU);
    EXPECT_EQ(count_e8m0_tensors(compiled.get_runtime_model()), 0u);
}

}  // namespace
