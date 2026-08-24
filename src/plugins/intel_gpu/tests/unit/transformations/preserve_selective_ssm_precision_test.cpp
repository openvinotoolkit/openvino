// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/transformations/preserve_selective_ssm_precision.hpp"

#include <gtest/gtest.h>

#include "openvino/op/paged_selective_ssm.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/selective_ssm.hpp"
#include "openvino/op/shape_of.hpp"
#include "transformations/rt_info/disable_precision_conversion.hpp"

namespace ov::intel_gpu::test {
namespace {

void expect_conversion_disabled(const std::shared_ptr<ov::Node>& node) {
    EXPECT_TRUE(ov::is_conversion_disabled(node, ov::element::dynamic, ov::element::dynamic));
}

TEST(PreserveSelectiveSSMPrecisionTest, MarksSelectiveSSMAndDataInputs) {
    const auto A = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{4});
    const auto dt = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 2, 4});
    const auto B = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 2, 2, 3});
    const auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 2, 4, 5});
    const auto C = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 2, 2, 3});
    const auto state = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 5, 3});
    const ov::ParameterVector parameters{A, dt, B, x, C, state};
    const auto ssm = std::make_shared<ov::op::internal::SelectiveSSM>(A, dt, B, x, C, state);
    const auto model = std::make_shared<ov::Model>(ssm->outputs(), parameters);

    PreserveSelectiveSSMPrecision pass;
    EXPECT_FALSE(pass.run_on_model(model));

    expect_conversion_disabled(ssm);
    for (const auto& parameter : parameters) {
        expect_conversion_disabled(parameter);
    }
}

TEST(PreserveSelectiveSSMPrecisionTest, MarksPagedOperationAndAllInputs) {
    const auto A = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{4});
    const auto dt = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 4});
    const auto B = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 2, 3});
    const auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 4, 5});
    const auto C = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 2, 3});
    const auto state = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 4, 5, 3});
    const auto subsequences = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{2});
    const auto blocks = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{2});
    const auto block_begins = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{2});
    const auto processed = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{1});
    const auto intervals = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{1});
    const ov::ParameterVector data_parameters{A, dt, B, x, C, state};
    const ov::ParameterVector metadata_parameters{subsequences, blocks, block_begins, processed, intervals};
    ov::ParameterVector parameters = data_parameters;
    parameters.insert(parameters.end(), metadata_parameters.begin(), metadata_parameters.end());
    const auto ssm = std::make_shared<ov::op::internal::PagedSelectiveSSM>(A, dt, B, x, C, state, subsequences, blocks, block_begins, processed, intervals);
    const auto model = std::make_shared<ov::Model>(ssm->outputs(), parameters);

    PreserveSelectiveSSMPrecision pass;
    EXPECT_FALSE(pass.run_on_model(model));

    expect_conversion_disabled(ssm);
    for (const auto& parameter : data_parameters) {
        expect_conversion_disabled(parameter);
    }
    for (const auto& parameter : metadata_parameters) {
        expect_conversion_disabled(parameter);
    }
}

TEST(PreserveSelectiveSSMPrecisionTest, EliminatesStaticallyEmptySelectiveSSM) {
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

TEST(PreserveSelectiveSSMPrecisionTest, AddsZeroCopyViewsForSingleDynamicOutputs) {
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

TEST(PreserveSelectiveSSMPrecisionTest, HandlesSingleStaticOutputs) {
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

TEST(PreserveSelectiveSSMPrecisionTest, DoesNotAddViewsWhenBothOutputsAreUsed) {
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
