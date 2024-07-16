// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/node_builders/constant.hpp"
#include "openvino/opsets/opset10.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "transformations/utils/utils.hpp"
#include "utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;
using namespace ov::test;

namespace CPUSubgraphTestsDefinitions {
template <typename NodeType, typename... Args>
static std::shared_ptr<ov::Node> make_layer_with_bias(Args&&... args) {
    const auto node = std::make_shared<NodeType>(std::forward<Args>(args)...);
    const auto& precision = node->get_output_element_type(0);
    const auto bias_const = ov::test::utils::make_constant(precision, ov::Shape{});
    const auto bias = std::make_shared<ov::opset10::Add>(node, bias_const);
    return bias;
}

/*
       Parameter(4D)
           |
        Reshape(3D)
           |
     Transpose(0, 2, 1)
           |
         MatMul
           |
     Transpose(0, 2, 1)
           |
        Reshape(4D)
           |
     GroupConvolution
           |
        Reshape(3D)
           |
     Transpose(0, 2, 1)
           |
         MatMul
*/

struct ExpectedResult {
    size_t expected_reshape_count;
    size_t expected_transpose_count;
    size_t expected_reorder_count;
};

using MergeTransposeReorderTestParams = std::tuple<InputShape, ExpectedResult>;
class MergeTransposeReorderCPUTest : public testing::WithParamInterface<MergeTransposeReorderTestParams>, virtual public SubgraphBaseTest, public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MergeTransposeReorderTestParams> &obj) {
        InputShape input_shape;
        ExpectedResult expected_result;
        std::tie(input_shape, expected_result) = obj.param;

        std::ostringstream results;
        results << "IS=(" << ov::test::utils::partialShape2str({input_shape.first}) << "_";
        results << ")_TS=(";
        for (const auto& static_shape : input_shape.second) {
            results << ov::test::utils::vec2str(static_shape) << "_";
        }
        results << ")_reshape_count=" << expected_result.expected_reshape_count;
        results << "_transpose_count=" << expected_result.expected_transpose_count;
        results << "_reorder_count=" << expected_result.expected_reorder_count;
        return results.str();
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;
        InputShape input_shape;
        std::tie(input_shape, m_expected_result) = this->GetParam();
        init_input_shapes({input_shape});

        const auto precision = ov::element::f32;
        const auto shapeof_subgraph_prc = ov::element::i32;
        OPENVINO_ASSERT(inputDynamicShapes[0].rank().is_static() && inputDynamicShapes[0].size() == 4, "initSubgraph: only 4D shapes are supported");
        OPENVINO_ASSERT(inputDynamicShapes[0][1].is_static(), "initSubgraph: only static channels dim is supported");

        const auto param = std::make_shared<ov::opset10::Parameter>(precision, inputDynamicShapes[0]);
        const auto reshape_const_1 = ov::opset10::Constant::create(shapeof_subgraph_prc, {3}, {0, 0, -1});
        const auto reshape_1 = std::make_shared<ov::opset10::Reshape>(param, reshape_const_1, true);

        const auto transpose_const_1 = ov::opset10::Constant::create(shapeof_subgraph_prc, {3}, {0, 2, 1});
        const auto transpose_1 = std::make_shared<ov::opset10::Transpose>(reshape_1, transpose_const_1);

        const size_t channels = inputDynamicShapes[0][1].get_length();
        const size_t fc_out_channels = 512;
        const auto fc_weights_1 = ov::test::utils::make_constant(precision, ov::Shape{fc_out_channels, channels});
        const auto fc_1 = make_layer_with_bias<ov::opset10::MatMul>(transpose_1, fc_weights_1, false, true);

        const auto transpose_const_2 = ov::opset10::Constant::create(shapeof_subgraph_prc, {3}, {0, 2, 1});
        const auto transpose_2 = std::make_shared<ov::opset10::Transpose>(fc_1, transpose_const_2);
        const auto spatial_dims = ov::op::util::node_to_get_shape_value_of_indices_from_shape_source(param, {2, 3}, {}, shapeof_subgraph_prc);
        const auto unchangable_dims = ov::opset10::Constant::create(shapeof_subgraph_prc, {2}, {0, 0});
        const auto reshape_const_2 = ov::op::util::make_try_fold<ov::opset10::Concat>(ov::OutputVector{unchangable_dims, spatial_dims}, 0);
        const auto reshape_2 = std::make_shared<ov::opset10::Reshape>(transpose_2, reshape_const_2, true);

        const auto conv_weights = ov::test::utils::make_constant(precision, ov::Shape{fc_out_channels, 1, 1, 3, 3});
        const auto conv_with_bias = make_layer_with_bias<ov::opset10::GroupConvolution>(reshape_2,
                                                                              conv_weights,
                                                                              ov::Strides{1, 1},
                                                                              ov::CoordinateDiff{1, 1},
                                                                              ov::CoordinateDiff{1, 1},
                                                                              ov::Strides{1, 1});
        // It's necessary to force acdb layout to be sure that the reorder, which changes dims order, will be inserted
        // (by default acdb layout is chosen only on >= AVX512 platforms)
        const auto conv = conv_with_bias->get_input_node_shared_ptr(0);
        const auto acdb_format = CPUTestUtils::cpu_memory_format_t::acdb;
        conv->get_rt_info() = makeCPUInfo({acdb_format}, {acdb_format}, {});

        const auto dim_h = ov::op::util::node_to_get_shape_value_of_indices_from_shape_source(param, {2}, {}, shapeof_subgraph_prc);
        const auto dim_w = ov::op::util::node_to_get_shape_value_of_indices_from_shape_source(param, {3}, {}, shapeof_subgraph_prc);
        const auto fused_spatial_dims = ov::op::util::make_try_fold<ov::opset10::Multiply>(dim_h, dim_w);
        const auto reshape_const_3 = ov::op::util::make_try_fold<ov::opset10::Concat>(ov::OutputVector{unchangable_dims, fused_spatial_dims}, 0);
        const auto reshape_3 = std::make_shared<ov::opset10::Reshape>(conv_with_bias, reshape_const_3, true);
        const auto transpose_const_3 = ov::opset10::Constant::create(shapeof_subgraph_prc, {3}, {0, 2, 1});
        const auto transpose_3 = std::make_shared<ov::opset10::Transpose>(reshape_3, transpose_const_3);

        const auto fc_weights_2 = ov::test::utils::make_constant(precision, ov::Shape{channels, fc_out_channels});
        const auto fc_2 = make_layer_with_bias<ov::opset10::MatMul>(transpose_3, fc_weights_2, false, true);
        function = std::make_shared<ov::Model>(fc_2, ov::ParameterVector{param}, "MergeTransposeReorderModel");
    }

    void validate_exec_graph() {
        CheckNumberOfNodesWithType(compiledModel, "Reshape", m_expected_result.expected_reshape_count);
        CheckNumberOfNodesWithType(compiledModel, "Transpose", m_expected_result.expected_transpose_count);
        CheckNumberOfNodesWithType(compiledModel, "Reorder", m_expected_result.expected_reorder_count);
    }

private:
    ExpectedResult m_expected_result;
};

TEST_P(MergeTransposeReorderCPUTest, CompareWithRefs) {
    run();
    validate_exec_graph();
}

namespace {
std::vector<InputShape> static_shapes = {
    InputShape{{}, {{1, 32, 16, 16}}},
};

#if defined(OPENVINO_ARCH_ARM)
const ExpectedResult successfull_fuse_result{1, 1, 3};
#else
const ExpectedResult successfull_fuse_result{1, 1, 2};
#endif

#if defined(OPENVINO_ARCH_ARM) || defined(OPENVINO_ARCH_ARM64)
const ExpectedResult unsuccessfull_fuse_result{3, 3, 3};
#else
const ExpectedResult unsuccessfull_fuse_result{3, 3, 2};
#endif

INSTANTIATE_TEST_SUITE_P(smoke_MergeTransposeReorder_static, MergeTransposeReorderCPUTest,
                        ::testing::Combine(::testing::ValuesIn(static_shapes),
                                           ::testing::Values(successfull_fuse_result)),
                        MergeTransposeReorderCPUTest::getTestCaseName);

std::vector<InputShape> dynamic_shapes = {
    InputShape{{-1, 32, -1, -1}, {{1, 32, 16, 16}}},
    InputShape{{-1, 32, 16, 16}, {{1, 32, 16, 16}}},
};

INSTANTIATE_TEST_SUITE_P(smoke_MergeTransposeReorder_dynamic, MergeTransposeReorderCPUTest,
                        ::testing::Combine(::testing::ValuesIn(dynamic_shapes),
                                           ::testing::Values(unsuccessfull_fuse_result)),
                        MergeTransposeReorderCPUTest::getTestCaseName);
} // namespace
} // namespace CPUSubgraphTestsDefinitions
