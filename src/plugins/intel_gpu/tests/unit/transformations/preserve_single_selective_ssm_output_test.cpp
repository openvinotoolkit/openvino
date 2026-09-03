// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/transformations/preserve_single_selective_ssm_output.hpp"

#include <gtest/gtest.h>

#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/selective_ssm.hpp"
#include "openvino/op/shape_of.hpp"

namespace ov::intel_gpu::test {
namespace {

TEST(PreserveSingleSelectiveSSMOutputTest, EliminatesStaticallyEmptySelectiveSSM) {
    const auto A = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{4});
    const auto dt = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 0, 4});
    const auto B = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 0, 2, 3});
    const auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 0, 4, 5});
    const auto C = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 0, 2, 3});
    const auto state = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 5, 3});
    const ov::ParameterVector parameters{A, dt, B, x, C, state};
    const auto ssm = std::make_shared<ov::op::internal::SelectiveSSM>(A, dt, B, x, C, state);
    const auto model = std::make_shared<ov::Model>(ssm->outputs(), parameters);

    EliminateEmptySelectiveSSM pass;
    EXPECT_TRUE(pass.run_on_model(model));
    EXPECT_EQ(model->get_results()[0]->input_value(0), x->output(0));
    EXPECT_EQ(model->get_results()[1]->input_value(0), state->output(0));
}

TEST(PreserveSingleSelectiveSSMOutputTest, AddsZeroCopyViewsForSingleDynamicOutputs) {
    const auto dynamic_1d = ov::PartialShape::dynamic(1);
    const auto dynamic_3d = ov::PartialShape::dynamic(3);
    const auto dynamic_4d = ov::PartialShape::dynamic(4);
    const auto A = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, dynamic_1d);
    const auto dt = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, dynamic_3d);
    const auto B = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, dynamic_4d);
    const auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, dynamic_4d);
    const auto C = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, dynamic_4d);
    const auto state = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, dynamic_4d);
    const ov::ParameterVector parameters{A, dt, B, x, C, state};
    for (size_t output_index : {0, 1}) {
        const auto ssm = std::make_shared<ov::op::internal::SelectiveSSM>(A, dt, B, x, C, state);
        const auto model = std::make_shared<ov::Model>(ov::OutputVector{ssm->output(output_index)}, parameters);

        PreserveSingleSelectiveSSMOutput pass;
        EXPECT_TRUE(pass.run_on_model(model));
        const auto output_view = ov::as_type_ptr<ov::op::v1::Reshape>(model->get_results()[0]->get_input_node_shared_ptr(0));
        ASSERT_NE(output_view, nullptr);
        EXPECT_EQ(output_view->input_value(0), ssm->output(output_index));
        const auto output_shape = ov::as_type_ptr<ov::op::v3::ShapeOf>(output_view->get_input_node_shared_ptr(1));
        ASSERT_NE(output_shape, nullptr);
        EXPECT_EQ(output_shape->input_value(0), ssm->output(output_index));
        EXPECT_EQ(ssm->output(output_index).get_target_inputs().size(), 2);
    }
}

TEST(PreserveSingleSelectiveSSMOutputTest, HandlesSingleStaticOutputs) {
    const auto A = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{4});
    const auto dt = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 3, 4});
    const auto B = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 3, 2, 5});
    const auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 3, 4, 6});
    const auto C = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 3, 2, 5});
    const auto state = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 6, 5});
    const ov::ParameterVector parameters{A, dt, B, x, C, state};
    for (size_t output_index : {0, 1}) {
        const auto ssm = std::make_shared<ov::op::internal::SelectiveSSM>(A, dt, B, x, C, state);
        const auto model = std::make_shared<ov::Model>(ov::OutputVector{ssm->output(output_index)}, parameters);

        PreserveSingleSelectiveSSMOutput pass;
        EXPECT_EQ(pass.run_on_model(model), output_index == 1);
        if (output_index == 0) {
            EXPECT_EQ(model->get_results()[0]->input_value(0), ssm->output(0));
            continue;
        }

        const auto output_view = ov::as_type_ptr<ov::op::v1::Reshape>(model->get_results()[0]->get_input_node_shared_ptr(0));
        ASSERT_NE(output_view, nullptr);
        EXPECT_EQ(output_view->input_value(0), ssm->output(1));
        const auto output_shape = ov::as_type_ptr<ov::op::v3::ShapeOf>(output_view->get_input_node_shared_ptr(1));
        ASSERT_NE(output_shape, nullptr);
        EXPECT_EQ(output_shape->input_value(0), ssm->output(1));
        EXPECT_EQ(ssm->output(1).get_target_inputs().size(), 2);
    }
}

TEST(PreserveSingleSelectiveSSMOutputTest, DoesNotAddViewsWhenBothOutputsAreUsed) {
    const auto A = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(1));
    const auto dt = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(3));
    const auto B = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
    const auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
    const auto C = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
    const auto state = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
    const ov::ParameterVector parameters{A, dt, B, x, C, state};
    const auto ssm = std::make_shared<ov::op::internal::SelectiveSSM>(A, dt, B, x, C, state);
    const auto model = std::make_shared<ov::Model>(ssm->outputs(), parameters);

    PreserveSingleSelectiveSSMOutput pass;
    EXPECT_FALSE(pass.run_on_model(model));
    EXPECT_EQ(model->get_results()[0]->input_value(0), ssm->output(0));
    EXPECT_EQ(model->get_results()[1]->input_value(0), ssm->output(1));
}

}  // namespace
}  // namespace ov::intel_gpu::test
