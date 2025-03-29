// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset5.hpp"

using namespace ov;

TEST(SmartReshapeTests, TensorIteratorStaticParameters) {
    std::shared_ptr<ov::Model> f(nullptr);
    {
        // That which we iterate over
        auto X = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1, 1});
        auto Y = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1, 1});
        auto M = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1, 1});

        // Set up the cell body, a function from (Xi, Yi) -> (Zo)
        // Body parameters
        auto Xi = std::make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());
        auto Yi = std::make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());
        auto M_body = std::make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());
        auto body_condition = std::make_shared<opset5::Constant>(element::boolean, Shape{}, true);

        // Body
        auto sum = std::make_shared<opset5::Add>(Xi, Yi);
        auto Zo = std::make_shared<opset5::Multiply>(sum, M_body);
        auto body = std::make_shared<ov::Model>(OutputVector{Zo, body_condition, sum}, ParameterVector{Xi, Yi, M_body});

        auto tensor_iterator = std::make_shared<opset5::TensorIterator>();
        tensor_iterator->set_function(body);

        tensor_iterator->set_sliced_input(Xi, X, 0, 1, 1, -1, 2);
        tensor_iterator->set_sliced_input(Yi, Y, -1, -1, 1, 0, 1);
        tensor_iterator->set_merged_input(M_body, M, Zo);

        // Output 0 is last Zo
        auto out0 = tensor_iterator->get_iter_value(body_condition, -1);
        auto out1 = tensor_iterator->get_iter_value(Zo, -1);
        // Output 1 is concat of Zos
        // start=0, stride=1, part_size=1, end=-1, axis=1
        auto out2 = tensor_iterator->get_concatenated_slices(Zo, 0, 1, 1, -1, 1);
        auto out3 = tensor_iterator->get_iter_value(sum, -1);

        f = std::make_shared<Model>(OutputVector{out0, out1, out2, out3}, ParameterVector{X, Y, M});
    }

    ASSERT_TRUE(f->get_results()[0]->get_output_partial_shape(0).compatible({}));
    ASSERT_TRUE(f->get_results()[1]->get_output_partial_shape(0).compatible({1, 1, 1}));
    // concat output (seq len = 1, so it means num_iter = 1)
    ASSERT_TRUE(f->get_results()[2]->get_output_partial_shape(0).compatible({1, 1, 1}));
    ASSERT_TRUE(f->get_results()[3]->get_output_partial_shape(0).compatible({1, 1, 1}));

    auto unh = std::make_shared<ov::pass::UniqueNamesHolder>();
    init_unique_names(f, unh);
    EXPECT_ANY_THROW(f->reshape({{f->get_parameters()[0]->get_friendly_name(), {32, 1, 10}},
                                 {f->get_parameters()[1]->get_friendly_name(), {32, 10, 1}},
                                 {f->get_parameters()[2]->get_friendly_name(), {32, 1, 10}}}));
}

TEST(SmartReshapeTests, TensorIteratorDynamicParameters) {
    std::shared_ptr<ov::Model> f(nullptr);
    {
        // That which we iterate over
        auto X = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1, 1});
        auto Y = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1, 1});
        auto M = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1, 1});

        // Set up the cell body, a function from (Xi, Yi) -> (Zo)
        // Body parameters
        auto Xi = std::make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());
        auto Yi = std::make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());
        auto M_body = std::make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());
        auto body_condition = std::make_shared<opset5::Constant>(element::boolean, Shape{}, true);

        // Body
        auto sum = std::make_shared<opset5::Add>(Xi, Yi);
        auto Zo = std::make_shared<opset5::Multiply>(sum, M_body);
        auto body = std::make_shared<ov::Model>(OutputVector{Zo, body_condition, sum}, ParameterVector{Xi, Yi, M_body});

        auto tensor_iterator = std::make_shared<opset5::TensorIterator>();
        tensor_iterator->set_function(body);

        tensor_iterator->set_sliced_input(Xi, X, 0, 1, 1, -1, 2);
        tensor_iterator->set_sliced_input(Yi, Y, -1, -1, 1, 0, 1);
        tensor_iterator->set_merged_input(M_body, M, Zo);

        // Output 0 is last Zo
        auto out0 = tensor_iterator->get_iter_value(body_condition, -1);
        auto out1 = tensor_iterator->get_iter_value(Zo, -1);
        // Output 1 is concat of Zos
        // start=0, stride=1, part_size=1, end=-1, axis=1
        auto out2 = tensor_iterator->get_concatenated_slices(Zo, 0, 1, 1, -1, 1);
        auto out3 = tensor_iterator->get_iter_value(sum, -1);

        f = std::make_shared<Model>(OutputVector{out0, out1, out2, out3}, ParameterVector{X, Y, M});
    }

    ASSERT_TRUE(f->get_results()[0]->get_output_partial_shape(0).compatible({}));
    ASSERT_TRUE(f->get_results()[1]->get_output_partial_shape(0).compatible({1, 1, 1}));
    // concat output (seq len = 1, so it means num_iter = 1)
    ASSERT_TRUE(f->get_results()[2]->get_output_partial_shape(0).compatible({1, 1, 1}));
    ASSERT_TRUE(f->get_results()[3]->get_output_partial_shape(0).compatible({1, 1, 1}));

    auto unh = std::make_shared<ov::pass::UniqueNamesHolder>();
    init_unique_names(f, unh);
    EXPECT_ANY_THROW(f->reshape({{f->get_parameters()[0]->get_friendly_name(), {32, 1, 10}},
                                 {f->get_parameters()[1]->get_friendly_name(), {32, 10, 1}},
                                 {f->get_parameters()[2]->get_friendly_name(), {32, 1, 10}}}));
}

TEST(SmartReshapeTests, LoopStaticParameters) {
    std::shared_ptr<ov::Model> f(nullptr);
    {
        // That which we iterate over
        auto X = std::make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());
        auto Y = std::make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());
        auto M = std::make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());

        // Set up the cell body, a function from (Xi, Yi) -> (Zo)
        // Body parameters
        auto current_iteration = std::make_shared<opset5::Parameter>(element::i64, Shape{});
        auto Xi = std::make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());
        auto Yi = std::make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());
        auto M_body = std::make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());
        auto body_condition = std::make_shared<opset5::Constant>(element::boolean, Shape{}, true);

        auto trip_count = std::make_shared<opset5::Constant>(element::i64, Shape{}, 10);
        auto exec_condition = std::make_shared<opset5::Constant>(element::boolean, Shape{}, true);
        // Body
        auto sum = std::make_shared<opset5::Add>(Xi, Yi);
        auto Zo = std::make_shared<opset5::Multiply>(sum, M_body);
        auto body = std::make_shared<ov::Model>(OutputVector{Zo, body_condition, sum},
                                                ParameterVector{Xi, current_iteration, Yi, M_body});

        auto loop = std::make_shared<opset5::Loop>(trip_count, exec_condition);
        loop->set_function(body);
        loop->set_special_body_ports(opset5::Loop::SpecialBodyPorts{1, 1});

        loop->set_sliced_input(Xi, X, 0, 1, 1, -1, 2);
        loop->set_sliced_input(Yi, Y, -1, -1, 1, 0, 1);
        loop->set_merged_input(M_body, M, Zo);

        // Output 0 is last Zo
        auto out0 = loop->get_iter_value(body_condition, -1);
        auto out1 = loop->get_iter_value(Zo, -1);
        // Output 1 is concat of Zos
        // start=0, stride=1, part_size=1, end=-1, axis=1
        auto out2 = loop->get_concatenated_slices(Zo, 0, 1, 1, -1, 1);
        auto out3 = loop->get_iter_value(sum, -1);

        f = std::make_shared<Model>(OutputVector{out0, out1, out2, out3}, ParameterVector{X, Y, M});
    }

    ASSERT_TRUE(f->get_results()[0]->get_output_partial_shape(0).compatible({}));
    ASSERT_TRUE(f->get_results()[1]->get_output_partial_shape(0).compatible(PartialShape::dynamic()));
    // concat output
    ASSERT_TRUE(f->get_results()[2]->get_output_partial_shape(0).compatible(PartialShape::dynamic()));
    ASSERT_TRUE(f->get_results()[3]->get_output_partial_shape(0).compatible(PartialShape::dynamic()));

    auto unh = std::make_shared<ov::pass::UniqueNamesHolder>();
    init_unique_names(f, unh);
    EXPECT_ANY_THROW(f->reshape({{f->get_parameters()[0]->get_friendly_name(), {32, 1, 10}},
                                 {f->get_parameters()[1]->get_friendly_name(), {32, 10, 1}},
                                 {f->get_parameters()[2]->get_friendly_name(), {32, 1, 10}}}));
}

TEST(SmartReshapeTests, LoopDynamicParameters) {
    std::shared_ptr<ov::Model> f(nullptr);
    {
        // That which we iterate over
        auto X = std::make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());
        auto Y = std::make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());
        auto M = std::make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());

        // Set up the cell body, a function from (Xi, Yi) -> (Zo)
        // Body parameters
        auto current_iteration = std::make_shared<opset5::Parameter>(element::i64, Shape{});
        auto Xi = std::make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());
        auto Yi = std::make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());
        auto M_body = std::make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());
        auto body_condition = std::make_shared<opset5::Constant>(element::boolean, Shape{}, true);

        auto trip_count = std::make_shared<opset5::Constant>(element::i64, Shape{}, 10);
        auto exec_condition = std::make_shared<opset5::Constant>(element::boolean, Shape{}, true);
        // Body
        auto sum = std::make_shared<opset5::Add>(Xi, Yi);
        auto Zo = std::make_shared<opset5::Multiply>(sum, M_body);
        auto body = std::make_shared<ov::Model>(OutputVector{Zo, body_condition, sum},
                                                ParameterVector{Xi, current_iteration, Yi, M_body});

        auto loop = std::make_shared<opset5::Loop>(trip_count, exec_condition);
        loop->set_function(body);
        loop->set_special_body_ports(opset5::Loop::SpecialBodyPorts{1, 1});

        loop->set_sliced_input(Xi, X, 0, 1, 1, -1, 2);
        loop->set_sliced_input(Yi, Y, -1, -1, 1, 0, 1);
        loop->set_merged_input(M_body, M, Zo);

        // Output 0 is last Zo
        auto out0 = loop->get_iter_value(body_condition, -1);
        auto out1 = loop->get_iter_value(Zo, -1);
        // Output 1 is concat of Zos
        // start=0, stride=1, part_size=1, end=-1, axis=1
        auto out2 = loop->get_concatenated_slices(Zo, 0, 1, 1, -1, 1);
        auto out3 = loop->get_iter_value(sum, -1);

        f = std::make_shared<Model>(OutputVector{out0, out1, out2, out3}, ParameterVector{X, Y, M});
    }

    ASSERT_TRUE(f->get_results()[0]->get_output_partial_shape(0).compatible({}));
    ASSERT_TRUE(f->get_results()[1]->get_output_partial_shape(0).compatible(PartialShape::dynamic()));
    // concat output
    ASSERT_TRUE(f->get_results()[2]->get_output_partial_shape(0).compatible(PartialShape::dynamic()));
    ASSERT_TRUE(f->get_results()[3]->get_output_partial_shape(0).compatible(PartialShape::dynamic()));

    auto unh = std::make_shared<ov::pass::UniqueNamesHolder>();
    init_unique_names(f, unh);
    EXPECT_ANY_THROW(f->reshape({{f->get_parameters()[0]->get_friendly_name(), {32, 1, 10}},
                                 {f->get_parameters()[1]->get_friendly_name(), {32, 10, 1}},
                                 {f->get_parameters()[2]->get_friendly_name(), {32, 1, 10}}}));
}

TEST(SmartReshapeTests, LoopParentParametersUsedInBody) {
    std::shared_ptr<ov::Model> f(nullptr);
    {
        // That which we iterate over
        auto X = std::make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());
        auto Y = std::make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());
        auto add_Y = std::make_shared<opset5::Add>(
            Y,
            std::make_shared<opset5::Constant>(element::f32, Shape{}, std::vector<float>{0.f}));
        auto M = std::make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());

        // Set up the cell body, a function from (Xi, add_Y) -> (Zo)
        // Body parameters
        auto current_iteration = std::make_shared<opset5::Parameter>(element::i64, Shape{});
        auto Xi = std::make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());
        auto Yi = std::make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());
        auto M_body = std::make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());
        auto body_condition = std::make_shared<opset5::Constant>(element::boolean, Shape{}, true);

        auto trip_count = std::make_shared<opset5::Constant>(element::i64, Shape{}, 10);
        auto exec_condition = std::make_shared<opset5::Constant>(element::boolean, Shape{}, true);
        // Body
        auto sum = std::make_shared<opset5::Add>(Xi, Yi);
        auto Zo = std::make_shared<opset5::Multiply>(sum, M_body);
        auto body = std::make_shared<ov::Model>(OutputVector{Zo, body_condition, sum},
                                                ParameterVector{Xi, current_iteration, Yi, M_body});

        auto loop = std::make_shared<opset5::Loop>(trip_count, exec_condition);
        loop->set_function(body);
        loop->set_special_body_ports(opset5::Loop::SpecialBodyPorts{1, 1});

        loop->set_sliced_input(Xi, X, 0, 1, 1, -1, 2);
        loop->set_merged_input(M_body, M, Zo);
        // Set invariant input which uses parameter from parent graph
        loop->set_invariant_input(Yi, add_Y);

        // Output 0 is last Zo
        auto out0 = loop->get_iter_value(body_condition, -1);
        auto out1 = loop->get_iter_value(Zo, -1);
        // Output 1 is concat of Zos
        // start=0, stride=1, part_size=1, end=-1, axis=1
        auto out2 = loop->get_concatenated_slices(Zo, 0, 1, 1, -1, 1);
        auto out3 = loop->get_iter_value(sum, -1);

        f = std::make_shared<Model>(OutputVector{out0, out1, out2, out3}, ParameterVector{X, Y, M});
    }

    ASSERT_TRUE(f->get_results()[0]->get_output_partial_shape(0).compatible({}));
    ASSERT_TRUE(f->get_results()[1]->get_output_partial_shape(0).compatible(PartialShape::dynamic()));
    // concat output
    ASSERT_TRUE(f->get_results()[2]->get_output_partial_shape(0).compatible(PartialShape::dynamic()));
    ASSERT_TRUE(f->get_results()[3]->get_output_partial_shape(0).compatible(PartialShape::dynamic()));

    auto unh = std::make_shared<ov::pass::UniqueNamesHolder>();
    init_unique_names(f, unh);
    EXPECT_ANY_THROW(f->reshape({{f->get_parameters()[0]->get_friendly_name(), {4, 3, 2}},
                                 {f->get_parameters()[1]->get_friendly_name(), {4, 3, 2}},
                                 {f->get_parameters()[2]->get_friendly_name(), {4, 3, 2}}}));
}

TEST(SmartReshapeTests, TensorIteratorParentParameterUsedInBody) {
    std::shared_ptr<ov::Model> f(nullptr);
    {
        // That which we iterate over
        auto X = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1, 1});
        auto Y = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1, 1});
        auto add_Y = std::make_shared<opset5::Add>(
            Y,
            std::make_shared<opset5::Constant>(element::f32, Shape{}, std::vector<float>{0.f}));
        auto M = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1, 1});

        // Set up the cell body, a function from (Xi, add_Y) -> (Zo)
        // Body parameters
        auto Xi = std::make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());
        auto Yi = std::make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());
        auto M_body = std::make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());
        auto body_condition = std::make_shared<opset5::Constant>(element::boolean, Shape{}, true);

        // Body
        auto sum = std::make_shared<opset5::Add>(Xi, Yi);
        auto Zo = std::make_shared<opset5::Multiply>(sum, M_body);
        auto body = std::make_shared<ov::Model>(OutputVector{Zo, body_condition, sum}, ParameterVector{Xi, Yi, M_body});

        auto tensor_iterator = std::make_shared<opset5::TensorIterator>();
        tensor_iterator->set_function(body);

        tensor_iterator->set_sliced_input(Xi, X, 0, 1, 1, -1, 2);
        tensor_iterator->set_merged_input(M_body, M, Zo);
        // Set invariant input which uses parameter from parent graph
        tensor_iterator->set_invariant_input(Yi, add_Y);

        // Output 0 is last Zo
        auto out0 = tensor_iterator->get_iter_value(body_condition, -1);
        auto out1 = tensor_iterator->get_iter_value(Zo, -1);
        // Output 1 is concat of Zos
        // start=0, stride=1, part_size=1, end=-1, axis=1
        auto out2 = tensor_iterator->get_concatenated_slices(Zo, 0, 1, 1, -1, 1);
        auto out3 = tensor_iterator->get_iter_value(sum, -1);

        f = std::make_shared<Model>(OutputVector{out0, out1, out2, out3}, ParameterVector{X, Y, M});
    }

    ASSERT_TRUE(f->get_results()[0]->get_output_partial_shape(0).compatible({}));
    ASSERT_TRUE(f->get_results()[1]->get_output_partial_shape(0).compatible({1, 1, 1}));
    // concat output (seq len = 1, so it means num_iter = 1)
    ASSERT_TRUE(f->get_results()[2]->get_output_partial_shape(0).compatible({1, 1, 1}));
    ASSERT_TRUE(f->get_results()[3]->get_output_partial_shape(0).compatible({1, 1, 1}));

    auto unh = std::make_shared<ov::pass::UniqueNamesHolder>();
    init_unique_names(f, unh);
    EXPECT_ANY_THROW(f->reshape({{f->get_parameters()[0]->get_friendly_name(), {32, 1, 10}},
                                 {f->get_parameters()[1]->get_friendly_name(), {1, 1, 1}},
                                 {f->get_parameters()[2]->get_friendly_name(), {32, 1, 10}}}));
}
