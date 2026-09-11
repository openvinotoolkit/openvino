// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "pass/gn_decomposition.hpp"
#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/group_normalization.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/subtract.hpp"
#include "snippets/op/convert_saturation.hpp"
#include "snippets/op/reduce.hpp"
#include "snippets/op/reshape.hpp"
#include "snippets/pass/gn_decomposition.hpp"
#include "subgraph_group_normalization.hpp"
#include "subgraph_lowered.hpp"

namespace ov {
namespace test {
namespace snippets {

std::string GNDecompositionTest::getTestCaseName(testing::TestParamInfo<GroupNormalizationParams> obj) {
    const auto& [input_shape, num_group, eps] = obj.param;
    std::ostringstream result;
    result << "IS=" << ov::test::utils::partialShape2str({input_shape}) << "_";
    result << "num_group=" << num_group << "_";
    result << "eps=" << eps;
    return result.str();
}

void GNDecompositionTest::SetUp() {
    LoweringTests::SetUp();

    const auto& [data_shape, num_group, eps] = this->GetParam();
    OPENVINO_ASSERT(data_shape.size() >= 2, "First input rank for group normalization op should be greater than 1");
    PartialShape scaleShiftShape = PartialShape{data_shape[1]};
    std::vector<PartialShape> input_shapes = { data_shape, scaleShiftShape, scaleShiftShape};
    snippets_model = std::make_shared<GroupNormalizationFunction>(input_shapes, num_group, eps);
}

TEST_P(GNDecompositionTest, GNDecomposition) {
    auto subgraph = getLoweredSubgraph(snippets_model->getOriginal());
    model = subgraph->body_ptr();
    model_ref = snippets_model->getLowered();
}

TEST_F(TransformationTestsF, GNDecompositionNonF32SharesCommonNodes) {
    const auto data = std::make_shared<op::v0::Parameter>(element::f16, Shape{1, 8, 2, 2});
    const auto scale = std::make_shared<op::v0::Parameter>(element::f32, Shape{8});
    const auto bias = std::make_shared<op::v0::Parameter>(element::f32, Shape{8});
    const auto group_norm = std::make_shared<op::v12::GroupNormalization>(data, scale, bias, 4, 0.0001);
    model = std::make_shared<Model>(OutputVector{group_norm}, ParameterVector{data, scale, bias});

    ov::pass::Manager decomposition_manager;
    decomposition_manager.register_pass<ov::snippets::pass::GNDecomposition>();
    decomposition_manager.run_passes(model);

    std::vector<std::shared_ptr<ov::snippets::op::ConvertSaturation>> data_converts;
    std::vector<std::shared_ptr<op::v0::Constant>> group_size_constants;
    for (const auto& node : model->get_ops()) {
        if (const auto convert = ov::as_type_ptr<ov::snippets::op::ConvertSaturation>(node)) {
            const auto reshape = convert->input_value(0).get_node_shared_ptr();
            if (ov::is_type<ov::snippets::op::Reshape>(reshape) &&
                reshape->input_value(0).get_node_shared_ptr() == data) {
                data_converts.push_back(convert);
            }
        }

        if (const auto constant = ov::as_type_ptr<op::v0::Constant>(node);
            constant && constant->get_shape() == Shape{} && constant->get_vector<float>().at(0) == 0.125F) {
            group_size_constants.push_back(constant);
        }
    }

    ASSERT_EQ(data_converts.size(), 1);
    const auto& convert_consumers = data_converts.front()->output(0).get_target_inputs();
    ASSERT_EQ(convert_consumers.size(), 2);
    bool has_reduce_sum = false;
    bool has_subtract = false;
    for (const auto& consumer : convert_consumers) {
        has_reduce_sum |= ov::is_type<ov::snippets::op::ReduceSum>(consumer.get_node());
        has_subtract |= ov::is_type<op::v1::Subtract>(consumer.get_node());
    }
    EXPECT_TRUE(has_reduce_sum);
    EXPECT_TRUE(has_subtract);

    ASSERT_EQ(group_size_constants.size(), 1);
    const auto& constant_consumers = group_size_constants.front()->output(0).get_target_inputs();
    ASSERT_EQ(constant_consumers.size(), 2);
    for (const auto& consumer : constant_consumers) {
        EXPECT_TRUE(ov::is_type<op::v1::Multiply>(consumer.get_node()));
        EXPECT_EQ(consumer.get_index(), 1);
    }
}

namespace GNDecompositionTestInstantiation {

const std::vector<ov::PartialShape> input_shapes{{1, 8},
                                                 {1, 8, 18},
                                                 {1, 16, 8, 5},
                                                 {3, 8, 2, 2, 3},
                                                 {3, 8, 2, 2, 3, 3}};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_GNDecomposition,
                         GNDecompositionTest,
                         ::testing::Combine(::testing::ValuesIn(input_shapes),
                                            ::testing::Values(4),
                                            ::testing::Values(0.0001)),
                         GNDecompositionTest::getTestCaseName);

}  // namespace GNDecompositionTestInstantiation
}  // namespace snippets
}  // namespace test
}  // namespace ov
