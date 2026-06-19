// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/loop.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/type_prop.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/tensor_iterator.hpp"
#include "openvino/op/unsqueeze.hpp"

using namespace std;
using namespace ov;

// trip_count = 10
// execution_condition = true
// body_condition = true
// all shapes are static, 10 iterations will be executed
TEST(type_prop, loop_operation_for_mode_10_iter_static_shapes) {
    // That which we iterate over
    auto X = make_shared<ov::op::v0::Parameter>(element::f32, Shape{32, 1, 10});
    auto Y = make_shared<ov::op::v0::Parameter>(element::f32, Shape{32, 1, 10});
    auto M = make_shared<ov::op::v0::Parameter>(element::f32, Shape{32, 1, 10});

    // Set up the cell body, a function from (Xi, Yi) -> (Zo)
    // Body parameters
    auto current_iteration = make_shared<ov::op::v0::Parameter>(element::i64, Shape{1});
    auto Xi = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto Yi = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto M_body = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto body_condition = std::make_shared<ov::op::v0::Constant>(element::boolean, Shape{1}, true);

    auto trip_count = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, 10);
    auto exec_condition = std::make_shared<ov::op::v0::Constant>(element::boolean, Shape{1}, true);
    // Body
    auto sum = make_shared<ov::op::v1::Add>(Xi, Yi);
    auto Zo = make_shared<ov::op::v1::Multiply>(sum, M_body);
    auto body =
        make_shared<Model>(OutputVector{body_condition, Zo}, ParameterVector{current_iteration, Xi, Yi, M_body});

    auto loop = make_shared<ov::op::v5::Loop>(trip_count, exec_condition);
    loop->set_function(body);

    loop->set_invariant_input(Xi, X);
    loop->set_invariant_input(Yi, Y);
    loop->set_merged_input(M_body, M, Zo);

    loop->set_special_body_ports(ov::op::v5::Loop::SpecialBodyPorts{-1, 0});

    // check input descriptors
    for (auto& desc : loop->get_input_descriptions()) {
        auto type_info = desc->get_type_info();
        if (std::strcmp(type_info.name, "InvariantInputDescription") == 0) {
            auto input_desc = ov::as_type_ptr<ov::op::v0::TensorIterator::InvariantInputDescription>(desc);
            EXPECT_NE(input_desc, nullptr);
        } else if (std::strcmp(type_info.name, "SliceInputDescription") == 0) {
            auto input_desc = ov::as_type_ptr<ov::op::v0::TensorIterator::SliceInputDescription>(desc);
            EXPECT_NE(input_desc, nullptr);
        } else if (std::strcmp(type_info.name, "MergedInputDescription") == 0) {
            auto input_desc = ov::as_type_ptr<ov::op::v0::TensorIterator::MergedInputDescription>(desc);
            EXPECT_NE(input_desc, nullptr);
        }
    }

    // Output 0 is last Zo
    auto out0 = loop->get_iter_value(body_condition, -1);
    auto out1 = loop->get_iter_value(Zo, -1);
    // Output 1 is concat of Zos
    // start=0, stride=1, part_size=1, end=-1, axis=1
    auto out2 = loop->get_concatenated_slices(Zo, 0, 1, 1, -1, 1);

    // check output descriptors
    for (auto& desc : loop->get_output_descriptions()) {
        auto type_info = desc->get_type_info();
        if (std::strcmp(type_info.name, "ConcatOutputDescription") == 0) {
            auto output_desc = ov::as_type_ptr<ov::op::v0::TensorIterator::ConcatOutputDescription>(desc);
            EXPECT_NE(output_desc, nullptr);
        } else if (std::strcmp(type_info.name, "BodyOutputDescription") == 0) {
            auto output_desc = ov::as_type_ptr<ov::op::v0::TensorIterator::BodyOutputDescription>(desc);
            EXPECT_NE(output_desc, nullptr);
        }
    }
    auto result0 = make_shared<ov::op::v0::Result>(out0);
    auto result1 = make_shared<ov::op::v0::Result>(out1);
    auto result2 = make_shared<ov::op::v0::Result>(out2);
    Shape out0_shape{1};
    Shape out1_shape{32, 1, 10};
    Shape out2_shape{32, 10, 10};

    auto results = ResultVector{result0, result1, result2};
    auto f = make_shared<Model>(results, ParameterVector{X, Y, M});
    EXPECT_EQ(result0->get_output_shape(0), out0_shape);
    EXPECT_EQ(result1->get_output_shape(0), out1_shape);
    EXPECT_EQ(result2->get_output_shape(0), out2_shape);

    EXPECT_EQ(loop->get_output_shape(0), out0_shape);
    EXPECT_EQ(loop->get_output_shape(1), out1_shape);
    EXPECT_EQ(loop->get_output_shape(2), out2_shape);
}

// trip_count = 10
// execution_condition = true
// body_condition = false
// will be executed only 1 iteration, all shapes are static
TEST(type_prop, loop_operation_dowhile_mode_1_iter_static_shapes) {
    // That which we iterate over
    auto X = make_shared<ov::op::v0::Parameter>(element::f32, Shape{32, 1, 10});
    auto Y = make_shared<ov::op::v0::Parameter>(element::f32, Shape{32, 1, 10});
    auto M = make_shared<ov::op::v0::Parameter>(element::f32, Shape{32, 1, 10});

    // Set up the cell body, a function from (Xi, Yi) -> (Zo)
    // Body parameters
    auto current_iteration = make_shared<ov::op::v0::Parameter>(element::i64, Shape{1});
    auto Xi = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto Yi = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto M_body = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto body_condition = std::make_shared<ov::op::v0::Constant>(element::boolean, Shape{1}, false);

    auto trip_count = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, 10);
    auto exec_condition = std::make_shared<ov::op::v0::Constant>(element::boolean, Shape{1}, true);
    // Body
    auto sum = make_shared<ov::op::v1::Add>(Xi, Yi);
    auto Zo = make_shared<ov::op::v1::Multiply>(sum, M_body);
    auto body =
        make_shared<Model>(OutputVector{body_condition, Zo}, ParameterVector{current_iteration, Xi, Yi, M_body});

    auto loop = make_shared<ov::op::v5::Loop>(trip_count, exec_condition);
    loop->set_function(body);
    loop->set_special_body_ports(ov::op::v5::Loop::SpecialBodyPorts{-1, 0});

    loop->set_invariant_input(Xi, X);
    loop->set_invariant_input(Yi, Y);
    loop->set_merged_input(M_body, M, Zo);

    // check input descriptors
    for (auto& desc : loop->get_input_descriptions()) {
        auto type_info = desc->get_type_info();
        if (std::strcmp(type_info.name, "InvariantInputDescription") == 0) {
            auto input_desc = ov::as_type_ptr<ov::op::v0::TensorIterator::InvariantInputDescription>(desc);
            EXPECT_NE(input_desc, nullptr);
        } else if (std::strcmp(type_info.name, "SliceInputDescription") == 0) {
            auto input_desc = ov::as_type_ptr<ov::op::v0::TensorIterator::SliceInputDescription>(desc);
            EXPECT_NE(input_desc, nullptr);
        } else if (std::strcmp(type_info.name, "MergedInputDescription") == 0) {
            auto input_desc = ov::as_type_ptr<ov::op::v0::TensorIterator::MergedInputDescription>(desc);
            EXPECT_NE(input_desc, nullptr);
        }
    }

    // Output 0 is last Zo
    auto out0 = loop->get_iter_value(body_condition, -1);
    auto out1 = loop->get_iter_value(Zo, -1);
    // Output 1 is concat of Zos
    // start=0, stride=1, part_size=1, end=-1, axis=1
    auto out2 = loop->get_concatenated_slices(Zo, 0, 1, 1, -1, 1);

    // check output descriptors
    for (auto& desc : loop->get_output_descriptions()) {
        auto type_info = desc->get_type_info();
        if (std::strcmp(type_info.name, "ConcatOutputDescription") == 0) {
            auto output_desc = ov::as_type_ptr<ov::op::v0::TensorIterator::ConcatOutputDescription>(desc);
            EXPECT_NE(output_desc, nullptr);
        } else if (std::strcmp(type_info.name, "BodyOutputDescription") == 0) {
            auto output_desc = ov::as_type_ptr<ov::op::v0::TensorIterator::BodyOutputDescription>(desc);
            EXPECT_NE(output_desc, nullptr);
        }
    }
    auto result0 = make_shared<ov::op::v0::Result>(out0);
    auto result1 = make_shared<ov::op::v0::Result>(out1);
    auto result2 = make_shared<ov::op::v0::Result>(out2);
    Shape out0_shape{1};
    Shape out1_shape{32, 1, 10};
    Shape out2_shape{32, 1, 10};

    auto results = ResultVector{result0, result1, result2};
    auto f = make_shared<Model>(results, ParameterVector{X, Y, M});
    EXPECT_EQ(result0->get_output_shape(0), out0_shape);
    EXPECT_EQ(result1->get_output_shape(0), out1_shape);
    EXPECT_EQ(result2->get_output_shape(0), out2_shape);

    EXPECT_EQ(loop->get_output_shape(0), out0_shape);
    EXPECT_EQ(loop->get_output_shape(1), out1_shape);
    EXPECT_EQ(loop->get_output_shape(2), out2_shape);
}

// trip_count = 10
// execution_condition = true
// body_condition is not a Constant
// concat output is not provided, another outputs will be static
TEST(type_prop, loop_operation_for_and_condition_mode_dynamic_iter_static_shapes) {
    // That which we iterate over
    auto X = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    auto Y = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    auto M = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});

    // Set up the cell body, a function from (Xi, Yi) -> (Zo)
    // Body parameters
    auto current_iteration = make_shared<ov::op::v0::Parameter>(element::i64, Shape{1});
    auto Xi = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto Yi = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto M_body = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto condition_const = std::make_shared<ov::op::v0::Constant>(element::f32, Shape{1}, 10);
    auto body_condition = std::make_shared<ov::op::v1::Greater>(M_body, condition_const);

    auto trip_count = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, 10);
    auto exec_condition = std::make_shared<ov::op::v0::Constant>(element::boolean, Shape{1}, true);
    // Body
    auto sum = make_shared<ov::op::v1::Add>(Xi, Yi);
    auto Zo = make_shared<ov::op::v1::Multiply>(sum, M_body);
    auto body = make_shared<Model>(OutputVector{body_condition, Zo}, ParameterVector{Xi, Yi, M_body});

    auto loop = make_shared<ov::op::v5::Loop>(trip_count, exec_condition);
    loop->set_function(body);
    loop->set_special_body_ports(ov::op::v5::Loop::SpecialBodyPorts{-1, 0});

    loop->set_invariant_input(Xi, X);
    loop->set_invariant_input(Yi, Y);
    loop->set_merged_input(M_body, M, Zo);

    // check input descriptors
    for (auto& desc : loop->get_input_descriptions()) {
        auto type_info = desc->get_type_info();
        if (std::strcmp(type_info.name, "InvariantInputDescription") == 0) {
            auto input_desc = ov::as_type_ptr<ov::op::v0::TensorIterator::InvariantInputDescription>(desc);
            EXPECT_NE(input_desc, nullptr);
        } else if (std::strcmp(type_info.name, "SliceInputDescription") == 0) {
            auto input_desc = ov::as_type_ptr<ov::op::v0::TensorIterator::SliceInputDescription>(desc);
            EXPECT_NE(input_desc, nullptr);
        } else if (std::strcmp(type_info.name, "MergedInputDescription") == 0) {
            auto input_desc = ov::as_type_ptr<ov::op::v0::TensorIterator::MergedInputDescription>(desc);
            EXPECT_NE(input_desc, nullptr);
        }
    }

    // Output 0 is last Zo
    auto out0 = loop->get_iter_value(body_condition, -1);
    auto out1 = loop->get_iter_value(Zo, -1);

    // check output descriptors
    for (auto& desc : loop->get_output_descriptions()) {
        auto type_info = desc->get_type_info();
        if (std::strcmp(type_info.name, "ConcatOutputDescription") == 0) {
            auto output_desc = ov::as_type_ptr<ov::op::v0::TensorIterator::ConcatOutputDescription>(desc);
            EXPECT_NE(output_desc, nullptr);
        } else if (std::strcmp(type_info.name, "BodyOutputDescription") == 0) {
            auto output_desc = ov::as_type_ptr<ov::op::v0::TensorIterator::BodyOutputDescription>(desc);
            EXPECT_NE(output_desc, nullptr);
        }
    }
    auto result0 = make_shared<ov::op::v0::Result>(out0);
    auto result1 = make_shared<ov::op::v0::Result>(out1);
    Shape out0_shape{1};
    Shape out1_shape{1};

    auto results = ResultVector{result0, result1};
    auto f = make_shared<Model>(results, ParameterVector{X, Y, M});
    EXPECT_EQ(result0->get_output_shape(0), out0_shape);
    EXPECT_EQ(result1->get_output_shape(0), out1_shape);

    EXPECT_EQ(loop->get_output_shape(0), out0_shape);
    EXPECT_EQ(loop->get_output_shape(1), out1_shape);
}

// trip_count = 10
// execution_condition = true
// body_condition is not a Constant
// concat output has only dynamic rank, another outputs are static
TEST(type_prop, loop_operation_for_and_condition_mode_dynamic_iter_dynamic_shapes) {
    // That which we iterate over
    auto X = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    auto Y = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    auto M = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});

    // Set up the cell body, a function from (Xi, Yi) -> (Zo)
    // Body parameters
    auto current_iteration = make_shared<ov::op::v0::Parameter>(element::i64, Shape{1});
    auto Xi = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto Yi = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto M_body = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto condition_const = std::make_shared<ov::op::v0::Constant>(element::f32, Shape{1}, 10);
    auto body_condition = std::make_shared<ov::op::v1::Greater>(M_body, condition_const);

    auto trip_count = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, 10);
    auto exec_condition = std::make_shared<ov::op::v0::Constant>(element::boolean, Shape{1}, true);
    // Body
    auto sum = make_shared<ov::op::v1::Add>(Xi, Yi);
    auto Zo = make_shared<ov::op::v1::Multiply>(sum, M_body);
    auto body =
        make_shared<Model>(OutputVector{body_condition, Zo}, ParameterVector{current_iteration, Xi, Yi, M_body});

    auto loop = make_shared<ov::op::v5::Loop>(trip_count, exec_condition);
    loop->set_function(body);
    loop->set_special_body_ports(ov::op::v5::Loop::SpecialBodyPorts{-1, 0});

    loop->set_invariant_input(Xi, X);
    loop->set_invariant_input(Yi, Y);
    loop->set_merged_input(M_body, M, Zo);

    // check input descriptors
    for (auto& desc : loop->get_input_descriptions()) {
        auto type_info = desc->get_type_info();
        if (std::strcmp(type_info.name, "InvariantInputDescription") == 0) {
            auto input_desc = ov::as_type_ptr<ov::op::v0::TensorIterator::InvariantInputDescription>(desc);
            EXPECT_NE(input_desc, nullptr);
        } else if (std::strcmp(type_info.name, "SliceInputDescription") == 0) {
            auto input_desc = ov::as_type_ptr<ov::op::v0::TensorIterator::SliceInputDescription>(desc);
            EXPECT_NE(input_desc, nullptr);
        } else if (std::strcmp(type_info.name, "MergedInputDescription") == 0) {
            auto input_desc = ov::as_type_ptr<ov::op::v0::TensorIterator::MergedInputDescription>(desc);
            EXPECT_NE(input_desc, nullptr);
        }
    }

    // Output 0 is last Zo
    auto out0 = loop->get_iter_value(body_condition, -1);
    auto out1 = loop->get_iter_value(Zo, -1);
    auto out2 = loop->get_concatenated_slices(Zo, 0, 1, 1, -1, 0);

    // check output descriptors
    for (auto& desc : loop->get_output_descriptions()) {
        auto type_info = desc->get_type_info();
        if (std::strcmp(type_info.name, "ConcatOutputDescription") == 0) {
            auto output_desc = ov::as_type_ptr<ov::op::v0::TensorIterator::ConcatOutputDescription>(desc);
            EXPECT_NE(output_desc, nullptr);
        } else if (std::strcmp(type_info.name, "BodyOutputDescription") == 0) {
            auto output_desc = ov::as_type_ptr<ov::op::v0::TensorIterator::BodyOutputDescription>(desc);
            EXPECT_NE(output_desc, nullptr);
        }
    }
    auto result0 = make_shared<ov::op::v0::Result>(out0);
    auto result1 = make_shared<ov::op::v0::Result>(out1);
    auto result2 = make_shared<ov::op::v0::Result>(out2);
    Shape out0_shape{1};
    Shape out1_shape{1};
    PartialShape out2_shape{PartialShape::dynamic(1)};

    auto results = ResultVector{result0, result1, result2};
    auto f = make_shared<Model>(results, ParameterVector{X, Y, M});
    EXPECT_EQ(result0->get_output_shape(0), out0_shape);
    EXPECT_EQ(result1->get_output_shape(0), out1_shape);
    EXPECT_EQ(result2->get_output_partial_shape(0), out2_shape);

    EXPECT_EQ(loop->get_output_shape(0), out0_shape);
    EXPECT_EQ(loop->get_output_shape(1), out1_shape);
    EXPECT_EQ(loop->get_output_partial_shape(2), out2_shape);
}

// trip_count = 10
// execution_condition = true
// body_condition is not a Constant
// inputs have dyanamic shape
// concat output has dynamic dimension on axis position, another outputs are static
TEST(type_prop, loop_operation_for_and_condition_mode_dynamic_iter_partially_dynamic_shapes) {
    // That which we iterate over
    auto X = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{Dimension::dynamic()});
    auto Y = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{Dimension::dynamic()});
    auto M = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});

    // Set up the cell body, a function from (Xi, Yi) -> (Zo)
    // Body parameters
    auto current_iteration = make_shared<ov::op::v0::Parameter>(element::i64, Shape{1});
    auto Xi = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto Yi = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto M_body = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto condition_const = std::make_shared<ov::op::v0::Constant>(element::f32, Shape{1}, 10);
    auto body_condition = std::make_shared<ov::op::v1::Greater>(M_body, condition_const);

    auto trip_count = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, 10);
    auto exec_condition = std::make_shared<ov::op::v0::Constant>(element::boolean, Shape{1}, true);
    // Body
    auto sum = make_shared<ov::op::v1::Add>(Xi, Yi);
    auto Zo = make_shared<ov::op::v1::Multiply>(sum, M_body);
    auto body =
        make_shared<Model>(OutputVector{body_condition, Zo}, ParameterVector{current_iteration, Xi, Yi, M_body});

    auto loop = make_shared<ov::op::v5::Loop>(trip_count, exec_condition);
    loop->set_function(body);
    loop->set_special_body_ports(ov::op::v5::Loop::SpecialBodyPorts{-1, 0});

    loop->set_invariant_input(Xi, X);
    loop->set_invariant_input(Yi, Y);
    loop->set_merged_input(M_body, M, Zo);

    // check input descriptors
    for (auto& desc : loop->get_input_descriptions()) {
        auto type_info = desc->get_type_info();
        if (std::strcmp(type_info.name, "InvariantInputDescription") == 0) {
            auto input_desc = ov::as_type_ptr<ov::op::v0::TensorIterator::InvariantInputDescription>(desc);
            EXPECT_NE(input_desc, nullptr);
        } else if (std::strcmp(type_info.name, "SliceInputDescription") == 0) {
            auto input_desc = ov::as_type_ptr<ov::op::v0::TensorIterator::SliceInputDescription>(desc);
            EXPECT_NE(input_desc, nullptr);
        } else if (std::strcmp(type_info.name, "MergedInputDescription") == 0) {
            auto input_desc = ov::as_type_ptr<ov::op::v0::TensorIterator::MergedInputDescription>(desc);
            EXPECT_NE(input_desc, nullptr);
        }
    }

    // Output 0 is last Zo
    auto out0 = loop->get_iter_value(body_condition, -1);
    auto out1 = loop->get_iter_value(Zo, -1);
    // axis=0 so sliced output on this dimension will be dynamic
    auto out2 = loop->get_concatenated_slices(Zo, 0, 1, 1, -1, 0);

    // check output descriptors
    for (auto& desc : loop->get_output_descriptions()) {
        auto type_info = desc->get_type_info();
        if (std::strcmp(type_info.name, "ConcatOutputDescription") == 0) {
            auto output_desc = ov::as_type_ptr<ov::op::v0::TensorIterator::ConcatOutputDescription>(desc);
            EXPECT_NE(output_desc, nullptr);
        } else if (std::strcmp(type_info.name, "BodyOutputDescription") == 0) {
            auto output_desc = ov::as_type_ptr<ov::op::v0::TensorIterator::BodyOutputDescription>(desc);
            EXPECT_NE(output_desc, nullptr);
        }
    }
    auto result0 = make_shared<ov::op::v0::Result>(out0);
    auto result1 = make_shared<ov::op::v0::Result>(out1);
    auto result2 = make_shared<ov::op::v0::Result>(out2);
    Shape out0_shape{1};
    PartialShape out1_shape{Dimension::dynamic()};
    PartialShape out2_shape{Dimension::dynamic()};

    auto results = ResultVector{result0, result1, result2};
    auto f = make_shared<Model>(results, ParameterVector{X, Y, M});
    EXPECT_EQ(result0->get_output_shape(0), out0_shape);
    EXPECT_EQ(result1->get_output_partial_shape(0), out1_shape);
    EXPECT_EQ(result2->get_output_partial_shape(0), out2_shape);

    EXPECT_EQ(loop->get_output_shape(0), out0_shape);
    EXPECT_EQ(loop->get_output_partial_shape(1), out1_shape);
    EXPECT_EQ(loop->get_output_partial_shape(2), out2_shape);
}

// trip_count = 10
// execution_condition = true
// body_condition is not a Constant
// inputs have partially known shape
// Axis of silced output is set as incorrect
TEST(type_prop, loop_operation_for_and_condition_mode_dynamic_iter_incorrect_sliced_output_axis) {
    // That which we iterate over
    auto X = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{1, 2, 3, Dimension::dynamic()});
    auto Y = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{1, 2, 3, Dimension::dynamic()});
    auto M = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});

    // Set up the cell body, a function from (Xi, Yi) -> (Zo)
    // Body parameters
    auto current_iteration = make_shared<ov::op::v0::Parameter>(element::i64, Shape{1});
    auto Xi = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto Yi = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto M_body = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto condition_const = std::make_shared<ov::op::v0::Constant>(element::f32, Shape{1}, 10);
    auto body_condition = std::make_shared<ov::op::v1::Greater>(M_body, condition_const);

    auto trip_count = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, 10);
    auto exec_condition = std::make_shared<ov::op::v0::Constant>(element::boolean, Shape{1}, true);
    // Body
    auto sum = make_shared<ov::op::v1::Add>(Xi, Yi);
    auto Zo = make_shared<ov::op::v1::Multiply>(sum, M_body);
    auto body =
        make_shared<Model>(OutputVector{body_condition, Zo}, ParameterVector{current_iteration, Xi, Yi, M_body});

    auto loop = make_shared<ov::op::v5::Loop>(trip_count, exec_condition);
    loop->set_function(body);
    loop->set_special_body_ports(ov::op::v5::Loop::SpecialBodyPorts{-1, 0});

    loop->set_invariant_input(Xi, X);
    loop->set_invariant_input(Yi, Y);
    loop->set_merged_input(M_body, M, Zo);

    const auto sliced_output_axis = 4;
    try {
        auto out = loop->get_concatenated_slices(Zo, 0, 1, 1, -1, sliced_output_axis);
        FAIL() << "Loop was created with incorrect axis of concatenated slices output.";
    } catch (const std::exception& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("out of the tensor rank range"));
    } catch (...) {
        FAIL() << "Construction loop operator failed for unexpected reason.";
    }
}

// trip_count = -1
// execution_condition = true
// body_condition = true
// concat output will be dynamic, another outputs are static
TEST(type_prop, loop_operation_infinite_loop_mode_dynamic_iter_dynamic_shapes) {
    // That which we iterate over
    auto X = make_shared<ov::op::v0::Parameter>(element::f32, Shape{32, 1, 10});
    auto Y = make_shared<ov::op::v0::Parameter>(element::f32, Shape{32, 1, 10});
    auto M = make_shared<ov::op::v0::Parameter>(element::f32, Shape{32, 1, 10});

    // Set up the cell body, a function from (Xi, Yi) -> (Zo)
    // Body parameters
    auto current_iteration = make_shared<ov::op::v0::Parameter>(element::i64, Shape{1});
    auto Xi = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto Yi = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto M_body = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto body_condition = std::make_shared<ov::op::v0::Constant>(element::boolean, Shape{1}, true);

    auto trip_count = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, -1);
    auto exec_condition = std::make_shared<ov::op::v0::Constant>(element::boolean, Shape{1}, true);
    // Body
    auto sum = make_shared<ov::op::v1::Add>(Xi, Yi);
    auto Zo = make_shared<ov::op::v1::Multiply>(sum, M_body);
    auto body =
        make_shared<Model>(OutputVector{body_condition, Zo}, ParameterVector{current_iteration, Xi, Yi, M_body});

    auto loop = make_shared<ov::op::v5::Loop>(trip_count, exec_condition);
    loop->set_function(body);
    loop->set_special_body_ports(ov::op::v5::Loop::SpecialBodyPorts{-1, 0});

    loop->set_invariant_input(Xi, X);
    loop->set_invariant_input(Yi, Y);
    loop->set_merged_input(M_body, M, Zo);

    // check input descriptors
    for (auto& desc : loop->get_input_descriptions()) {
        auto type_info = desc->get_type_info();
        if (std::strcmp(type_info.name, "InvariantInputDescription") == 0) {
            auto input_desc = ov::as_type_ptr<ov::op::v0::TensorIterator::InvariantInputDescription>(desc);
            EXPECT_NE(input_desc, nullptr);
        } else if (std::strcmp(type_info.name, "SliceInputDescription") == 0) {
            auto input_desc = ov::as_type_ptr<ov::op::v0::TensorIterator::SliceInputDescription>(desc);
            EXPECT_NE(input_desc, nullptr);
        } else if (std::strcmp(type_info.name, "MergedInputDescription") == 0) {
            auto input_desc = ov::as_type_ptr<ov::op::v0::TensorIterator::MergedInputDescription>(desc);
            EXPECT_NE(input_desc, nullptr);
        }
    }

    // Output 0 is last Zo
    auto out0 = loop->get_iter_value(body_condition, -1);
    auto out1 = loop->get_iter_value(Zo, -1);
    // Output 1 is concat of Zos
    // start=0, stride=1, part_size=1, end=-1, axis=1
    auto out2 = loop->get_concatenated_slices(Zo, 0, 1, 1, -1, 1);

    // check output descriptors
    for (auto& desc : loop->get_output_descriptions()) {
        auto type_info = desc->get_type_info();
        if (std::strcmp(type_info.name, "ConcatOutputDescription") == 0) {
            auto output_desc = ov::as_type_ptr<ov::op::v0::TensorIterator::ConcatOutputDescription>(desc);
            EXPECT_NE(output_desc, nullptr);
        } else if (std::strcmp(type_info.name, "BodyOutputDescription") == 0) {
            auto output_desc = ov::as_type_ptr<ov::op::v0::TensorIterator::BodyOutputDescription>(desc);
            EXPECT_NE(output_desc, nullptr);
        }
    }
    auto result0 = make_shared<ov::op::v0::Result>(out0);
    auto result1 = make_shared<ov::op::v0::Result>(out1);
    auto result2 = make_shared<ov::op::v0::Result>(out2);
    Shape out0_shape{1};
    Shape out1_shape{32, 1, 10};
    PartialShape out2_shape{32, Dimension::dynamic(), 10};

    auto results = ResultVector{result0, result1, result2};
    auto f = make_shared<Model>(results, ParameterVector{X, Y, M});
    EXPECT_EQ(result0->get_output_shape(0), out0_shape);
    EXPECT_EQ(result1->get_output_shape(0), out1_shape);
    EXPECT_EQ(result2->get_output_partial_shape(0), out2_shape);

    EXPECT_EQ(loop->get_output_shape(0), out0_shape);
    EXPECT_EQ(loop->get_output_shape(1), out1_shape);
    EXPECT_EQ(loop->get_output_partial_shape(2), out2_shape);
}

// SpecialBodyPorts (1, 1) <- test specific
// trip_count = 10
// execution_condition = true
// body_condition = true
// all shapes are static, 10 iterations will be executed
TEST(type_prop, loop_operation_for_mode_10_iter_static_shapes_special_body_ports) {
    // That which we iterate over
    auto X = make_shared<ov::op::v0::Parameter>(element::f32, Shape{32, 1, 10});
    auto Y = make_shared<ov::op::v0::Parameter>(element::f32, Shape{32, 1, 10});
    auto M = make_shared<ov::op::v0::Parameter>(element::f32, Shape{32, 1, 10});

    // Set up the cell body, a function from (Xi, Yi) -> (Zo)
    // Body parameters
    auto current_iteration = make_shared<ov::op::v0::Parameter>(element::i64, Shape{1});
    auto Xi = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto Yi = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto M_body = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto body_condition = std::make_shared<ov::op::v0::Constant>(element::boolean, Shape{1}, true);

    auto trip_count = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, 10);
    auto exec_condition = std::make_shared<ov::op::v0::Constant>(element::boolean, Shape{1}, true);
    // Body
    auto sum = make_shared<ov::op::v1::Add>(Xi, Yi);
    auto Zo = make_shared<ov::op::v1::Multiply>(sum, M_body);
    auto body =
        make_shared<Model>(OutputVector{Zo, body_condition}, ParameterVector{Xi, current_iteration, Yi, M_body});

    auto loop = make_shared<ov::op::v5::Loop>(trip_count, exec_condition);
    loop->set_function(body);
    loop->set_special_body_ports(ov::op::v5::Loop::SpecialBodyPorts{1, 1});

    loop->set_invariant_input(Xi, X);
    loop->set_invariant_input(Yi, Y);
    loop->set_merged_input(M_body, M, Zo);

    // check input descriptors
    for (auto& desc : loop->get_input_descriptions()) {
        auto type_info = desc->get_type_info();
        if (std::strcmp(type_info.name, "InvariantInputDescription") == 0) {
            auto input_desc = ov::as_type_ptr<ov::op::v0::TensorIterator::InvariantInputDescription>(desc);
            EXPECT_NE(input_desc, nullptr);
        } else if (std::strcmp(type_info.name, "SliceInputDescription") == 0) {
            auto input_desc = ov::as_type_ptr<ov::op::v0::TensorIterator::SliceInputDescription>(desc);
            EXPECT_NE(input_desc, nullptr);
        } else if (std::strcmp(type_info.name, "MergedInputDescription") == 0) {
            auto input_desc = ov::as_type_ptr<ov::op::v0::TensorIterator::MergedInputDescription>(desc);
            EXPECT_NE(input_desc, nullptr);
        }
    }

    // Output 0 is last Zo
    auto out0 = loop->get_iter_value(body_condition, -1);
    auto out1 = loop->get_iter_value(Zo, -1);
    // Output 1 is concat of Zos
    // start=0, stride=1, part_size=1, end=-1, axis=1
    auto out2 = loop->get_concatenated_slices(Zo, 0, 1, 1, -1, 1);

    // check output descriptors
    for (auto& desc : loop->get_output_descriptions()) {
        auto type_info = desc->get_type_info();
        if (std::strcmp(type_info.name, "ConcatOutputDescription") == 0) {
            auto output_desc = ov::as_type_ptr<ov::op::v0::TensorIterator::ConcatOutputDescription>(desc);
            EXPECT_NE(output_desc, nullptr);
        } else if (std::strcmp(type_info.name, "BodyOutputDescription") == 0) {
            auto output_desc = ov::as_type_ptr<ov::op::v0::TensorIterator::BodyOutputDescription>(desc);
            EXPECT_NE(output_desc, nullptr);
        }
    }
    auto result0 = make_shared<ov::op::v0::Result>(out0);
    auto result1 = make_shared<ov::op::v0::Result>(out1);
    auto result2 = make_shared<ov::op::v0::Result>(out2);
    Shape out0_shape{1};
    Shape out1_shape{32, 1, 10};
    Shape out2_shape{32, 10, 10};

    auto results = ResultVector{result0, result1, result2};
    auto f = make_shared<Model>(results, ParameterVector{X, Y, M});
    EXPECT_EQ(result0->get_output_shape(0), out0_shape);
    EXPECT_EQ(result1->get_output_shape(0), out1_shape);
    EXPECT_EQ(result2->get_output_shape(0), out2_shape);

    EXPECT_EQ(loop->get_output_shape(0), out0_shape);
    EXPECT_EQ(loop->get_output_shape(1), out1_shape);
    EXPECT_EQ(loop->get_output_shape(2), out2_shape);
}

// Scalars instead of 1d tensors with 1 element <-- test specific
// trip_count = 10
// execution_condition = true
// body_condition = true
// all shapes are static, 10 iterations will be executed
TEST(type_prop, loop_operation_for_mode_10_iter_static_shapes_special_body_ports_scalars) {
    // That which we iterate over
    auto X = make_shared<ov::op::v0::Parameter>(element::f32, Shape{32, 1, 10});
    auto Y = make_shared<ov::op::v0::Parameter>(element::f32, Shape{32, 1, 10});
    auto M = make_shared<ov::op::v0::Parameter>(element::f32, Shape{32, 1, 10});

    // Set up the cell body, a function from (Xi, Yi) -> (Zo)
    // Body parameters
    auto current_iteration = make_shared<ov::op::v0::Parameter>(element::i64, Shape{});
    auto Xi = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto Yi = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto M_body = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto body_condition = std::make_shared<ov::op::v0::Constant>(element::boolean, Shape{}, true);

    auto trip_count = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{}, 10);
    auto exec_condition = std::make_shared<ov::op::v0::Constant>(element::boolean, Shape{}, true);
    // Body
    auto sum = make_shared<ov::op::v1::Add>(Xi, Yi);
    auto Zo = make_shared<ov::op::v1::Multiply>(sum, M_body);
    auto body =
        make_shared<Model>(OutputVector{Zo, body_condition}, ParameterVector{Xi, current_iteration, Yi, M_body});

    auto loop = make_shared<ov::op::v5::Loop>(trip_count, exec_condition);
    loop->set_function(body);
    loop->set_special_body_ports(ov::op::v5::Loop::SpecialBodyPorts{1, 1});

    loop->set_invariant_input(Xi, X);
    loop->set_invariant_input(Yi, Y);
    loop->set_merged_input(M_body, M, Zo);

    // check input descriptors
    for (auto& desc : loop->get_input_descriptions()) {
        auto type_info = desc->get_type_info();
        if (std::strcmp(type_info.name, "InvariantInputDescription") == 0) {
            auto input_desc = ov::as_type_ptr<ov::op::v0::TensorIterator::InvariantInputDescription>(desc);
            EXPECT_NE(input_desc, nullptr);
        } else if (std::strcmp(type_info.name, "SliceInputDescription") == 0) {
            auto input_desc = ov::as_type_ptr<ov::op::v0::TensorIterator::SliceInputDescription>(desc);
            EXPECT_NE(input_desc, nullptr);
        } else if (std::strcmp(type_info.name, "MergedInputDescription") == 0) {
            auto input_desc = ov::as_type_ptr<ov::op::v0::TensorIterator::MergedInputDescription>(desc);
            EXPECT_NE(input_desc, nullptr);
        }
    }

    // Output 0 is last Zo
    auto out0 = loop->get_iter_value(body_condition, -1);
    auto out1 = loop->get_iter_value(Zo, -1);
    // Output 1 is concat of Zos
    // start=0, stride=1, part_size=1, end=-1, axis=1
    auto out2 = loop->get_concatenated_slices(Zo, 0, 1, 1, -1, 1);

    // check output descriptors
    for (auto& desc : loop->get_output_descriptions()) {
        auto type_info = desc->get_type_info();
        if (std::strcmp(type_info.name, "ConcatOutputDescription") == 0) {
            auto output_desc = ov::as_type_ptr<ov::op::v0::TensorIterator::ConcatOutputDescription>(desc);
            EXPECT_NE(output_desc, nullptr);
        } else if (std::strcmp(type_info.name, "BodyOutputDescription") == 0) {
            auto output_desc = ov::as_type_ptr<ov::op::v0::TensorIterator::BodyOutputDescription>(desc);
            EXPECT_NE(output_desc, nullptr);
        }
    }
    auto result0 = make_shared<ov::op::v0::Result>(out0);
    auto result1 = make_shared<ov::op::v0::Result>(out1);
    auto result2 = make_shared<ov::op::v0::Result>(out2);
    Shape out0_shape{};
    Shape out1_shape{32, 1, 10};
    Shape out2_shape{32, 10, 10};

    auto results = ResultVector{result0, result1, result2};
    auto f = make_shared<Model>(results, ParameterVector{X, Y, M});
    EXPECT_EQ(result0->get_output_shape(0), out0_shape);
    EXPECT_EQ(result1->get_output_shape(0), out1_shape);
    EXPECT_EQ(result2->get_output_shape(0), out2_shape);

    EXPECT_EQ(loop->get_output_shape(0), out0_shape);
    EXPECT_EQ(loop->get_output_shape(1), out1_shape);
    EXPECT_EQ(loop->get_output_shape(2), out2_shape);
}

// SliceInputs testing
// trip_count = 10
// execution_condition = true
// body_condition = true
// all shapes are static, 10 iterations will be executed
TEST(type_prop, loop_operation_10_iter_static_shapes_sliced_inputs) {
    // That which we iterate over
    auto X = make_shared<ov::op::v0::Parameter>(element::f32, Shape{32, 1, 10});
    auto Y = make_shared<ov::op::v0::Parameter>(element::f32, Shape{32, 10, 1});
    auto M = make_shared<ov::op::v0::Parameter>(element::f32, Shape{32, 1, 10});

    // Set up the cell body, a function from (Xi, Yi) -> (Zo)
    // Body parameters
    auto current_iteration = make_shared<ov::op::v0::Parameter>(element::i64, Shape{});
    auto Xi = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto Yi = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto M_body = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto body_condition = std::make_shared<ov::op::v0::Constant>(element::boolean, Shape{}, true);

    auto trip_count = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{}, 10);
    auto exec_condition = std::make_shared<ov::op::v0::Constant>(element::boolean, Shape{}, true);
    // Body
    auto sum = make_shared<ov::op::v1::Add>(Xi, Yi);
    auto Zo = make_shared<ov::op::v1::Multiply>(sum, M_body);
    auto body =
        make_shared<Model>(OutputVector{Zo, body_condition, sum}, ParameterVector{Xi, current_iteration, Yi, M_body});

    auto loop = make_shared<ov::op::v5::Loop>(trip_count, exec_condition);
    loop->set_function(body);
    loop->set_special_body_ports(ov::op::v5::Loop::SpecialBodyPorts{1, 1});

    loop->set_sliced_input(Xi, X, 0, 1, 1, -1, 2);
    loop->set_sliced_input(Yi, Y, -1, -1, 1, 0, 1);
    loop->set_merged_input(M_body, M, Zo);

    // check input descriptors
    for (auto& desc : loop->get_input_descriptions()) {
        auto type_info = desc->get_type_info();
        if (std::strcmp(type_info.name, "InvariantInputDescription") == 0) {
            auto input_desc = ov::as_type_ptr<ov::op::v0::TensorIterator::InvariantInputDescription>(desc);
            EXPECT_NE(input_desc, nullptr);
        } else if (std::strcmp(type_info.name, "SliceInputDescription") == 0) {
            auto input_desc = ov::as_type_ptr<ov::op::v0::TensorIterator::SliceInputDescription>(desc);
            EXPECT_NE(input_desc, nullptr);
        } else if (std::strcmp(type_info.name, "MergedInputDescription") == 0) {
            auto input_desc = ov::as_type_ptr<ov::op::v0::TensorIterator::MergedInputDescription>(desc);
            EXPECT_NE(input_desc, nullptr);
        }
    }

    // Output 0 is last Zo
    auto out0 = loop->get_iter_value(body_condition, -1);
    auto out1 = loop->get_iter_value(Zo, -1);
    // Output 1 is concat of Zos
    // start=0, stride=1, part_size=1, end=-1, axis=1
    auto out2 = loop->get_concatenated_slices(Zo, 0, 1, 1, -1, 1);
    auto out3 = loop->get_iter_value(sum, -1);

    // check output descriptors
    for (auto& desc : loop->get_output_descriptions()) {
        auto type_info = desc->get_type_info();
        if (std::strcmp(type_info.name, "ConcatOutputDescription") == 0) {
            auto output_desc = ov::as_type_ptr<ov::op::v0::TensorIterator::ConcatOutputDescription>(desc);
            EXPECT_NE(output_desc, nullptr);
        } else if (std::strcmp(type_info.name, "BodyOutputDescription") == 0) {
            auto output_desc = ov::as_type_ptr<ov::op::v0::TensorIterator::BodyOutputDescription>(desc);
            EXPECT_NE(output_desc, nullptr);
        }
    }
    auto result0 = make_shared<ov::op::v0::Result>(out0);
    auto result1 = make_shared<ov::op::v0::Result>(out1);
    auto result2 = make_shared<ov::op::v0::Result>(out2);
    auto result3 = make_shared<ov::op::v0::Result>(out3);
    Shape out0_shape{};
    Shape out1_shape{32, 1, 10};
    Shape out2_shape{32, 10, 10};
    Shape out3_shape{32, 1, 1};

    auto results = ResultVector{result0, result1, result2, result3};
    auto f = make_shared<Model>(results, ParameterVector{X, Y, M});
    EXPECT_EQ(result0->get_output_shape(0), out0_shape);
    EXPECT_EQ(result1->get_output_shape(0), out1_shape);
    EXPECT_EQ(result2->get_output_shape(0), out2_shape);
    EXPECT_EQ(result3->get_output_shape(0), out3_shape);

    EXPECT_EQ(loop->get_output_shape(0), out0_shape);
    EXPECT_EQ(loop->get_output_shape(1), out1_shape);
    EXPECT_EQ(loop->get_output_shape(2), out2_shape);
    EXPECT_EQ(loop->get_output_shape(3), out3_shape);
}

// SliceInputs testing
// trip_count = dynamic
// execution_condition = true
// body_condition = true
// input and output shapes has one dynamic dimension, other shapes are static, unknown iterations
// count will be executed
TEST(type_prop, loop_operation_dynamic_iter_dynamic_batch_shapes_sliced_inputs_concatenated_outputs) {
    // That which we iterate over
    auto X = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 1, 10});
    auto Y = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{32, Dimension::dynamic(), 10});
    auto M = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{32, 1, 10});
    auto T = make_shared<ov::op::v0::Parameter>(element::i64, Shape{});

    // Set up the cell body, a function from (Xi, Yi) -> (Zo)
    // Body parameters
    auto current_iteration = make_shared<ov::op::v0::Parameter>(element::i64, Shape{});
    auto Xi = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto Yi = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto M_body = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());

    auto body_condition = make_shared<ov::op::v0::Constant>(element::boolean, Shape{}, true);
    auto exec_condition = make_shared<ov::op::v0::Constant>(element::boolean, Shape{}, true);

    // Body
    auto sum = make_shared<ov::op::v1::Add>(Xi, Yi);
    auto Zo = make_shared<ov::op::v1::Multiply>(sum, M_body);
    auto body =
        make_shared<Model>(OutputVector{Zo, body_condition, sum}, ParameterVector{Xi, Yi, current_iteration, M_body});

    auto loop = make_shared<ov::op::v5::Loop>(T, exec_condition);
    loop->set_function(body);
    loop->set_special_body_ports(ov::op::v5::Loop::SpecialBodyPorts{2, 1});

    loop->set_sliced_input(Xi, X, 0, 1, 1, -1, 0);
    loop->set_sliced_input(Yi, Y, -1, -1, 1, 0, 1);
    loop->set_merged_input(M_body, M, Zo);

    // check input descriptors
    for (auto& desc : loop->get_input_descriptions()) {
        auto type_info = desc->get_type_info();
        if (std::strcmp(type_info.name, "InvariantInputDescription") == 0) {
            auto input_desc = ov::as_type_ptr<ov::op::v0::TensorIterator::InvariantInputDescription>(desc);
            EXPECT_NE(input_desc, nullptr);
        } else if (std::strcmp(type_info.name, "SliceInputDescription") == 0) {
            auto input_desc = ov::as_type_ptr<ov::op::v0::TensorIterator::SliceInputDescription>(desc);
            EXPECT_NE(input_desc, nullptr);
        } else if (std::strcmp(type_info.name, "MergedInputDescription") == 0) {
            auto input_desc = ov::as_type_ptr<ov::op::v0::TensorIterator::MergedInputDescription>(desc);
            EXPECT_NE(input_desc, nullptr);
        }
    }

    // Output 0 is last Zo
    auto out0 = loop->get_iter_value(body_condition, -1);
    auto out1 = loop->get_iter_value(Zo, -1);
    // Output 1 is concat of Zos
    // start=0, stride=1, part_size=1, end=-1, axis=1
    auto out2 = loop->get_concatenated_slices(Zo, 0, 1, 1, -1, 1);
    auto out3 = loop->get_iter_value(sum, -1);

    // check output descriptors
    for (auto& desc : loop->get_output_descriptions()) {
        auto type_info = desc->get_type_info();
        if (std::strcmp(type_info.name, "ConcatOutputDescription") == 0) {
            auto output_desc = ov::as_type_ptr<ov::op::v0::TensorIterator::ConcatOutputDescription>(desc);
            EXPECT_NE(output_desc, nullptr);
        } else if (std::strcmp(type_info.name, "BodyOutputDescription") == 0) {
            auto output_desc = ov::as_type_ptr<ov::op::v0::TensorIterator::BodyOutputDescription>(desc);
            EXPECT_NE(output_desc, nullptr);
        }
    }
    auto result0 = make_shared<ov::op::v0::Result>(out0);
    auto result1 = make_shared<ov::op::v0::Result>(out1);
    auto result2 = make_shared<ov::op::v0::Result>(out2);
    auto result3 = make_shared<ov::op::v0::Result>(out3);
    Shape out0_shape{};
    Shape out1_shape{32, 1, 10};
    PartialShape out2_shape{32, Dimension::dynamic(), 10};
    Shape out3_shape{32, 1, 10};

    auto results = ResultVector{result0, result1, result2, result3};
    auto f = make_shared<Model>(results, ParameterVector{X, Y, T, M});
    EXPECT_EQ(f->get_output_size(), 4);
    EXPECT_EQ(result0->get_output_shape(0), out0_shape);
    EXPECT_EQ(result1->get_output_shape(0), out1_shape);
    EXPECT_EQ(result2->get_output_partial_shape(0), out2_shape);
    EXPECT_EQ(result3->get_output_shape(0), out3_shape);

    const auto inp0_shape = Shape{1, 1, 10};
    const auto inp1_shape = Shape{32, 1, 10};
    const auto inp2_shape = Shape{};
    const auto inp3_shape = Shape{32, 1, 10};
    EXPECT_EQ(body->get_parameters().size(), 4);
    EXPECT_EQ(body->get_parameters().at(0)->get_shape(), inp0_shape);
    EXPECT_EQ(body->get_parameters().at(1)->get_shape(), inp1_shape);
    EXPECT_EQ(body->get_parameters().at(2)->get_shape(), inp2_shape);
    EXPECT_EQ(body->get_parameters().at(3)->get_shape(), inp3_shape);

    EXPECT_EQ(loop->get_output_size(), 4);
    EXPECT_EQ(loop->get_output_shape(0), out0_shape);
    EXPECT_EQ(loop->get_output_shape(1), out1_shape);
    EXPECT_EQ(loop->get_output_partial_shape(2), out2_shape);
    EXPECT_EQ(loop->get_output_shape(3), out3_shape);
}

// SliceInputs testing
// trip_count = dynamic
// execution_condition = true
// body_condition = true
// input and output shapes has one dynamic dimension, other shapes are static, unknown iterations
// count will be executed
TEST(type_prop, loop_operation_dynamic_iter_dynamic_shapes_sliced_inputs_concatenated_outputs) {
    // That which we iterate over
    auto X =
        make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{Dimension::dynamic(), Dimension::dynamic(), 10});
    auto T = make_shared<ov::op::v0::Parameter>(element::i64, Shape{});

    // Set up the cell body, a function from (Xi, Yi) -> (Zo)
    // Body parameters
    auto current_iteration = make_shared<ov::op::v0::Parameter>(element::i64, Shape{});
    auto Xi = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());

    auto body_condition = make_shared<ov::op::v0::Constant>(element::boolean, Shape{}, true);
    auto exec_condition = make_shared<ov::op::v0::Constant>(element::boolean, Shape{}, true);

    // Body
    auto Zo = make_shared<ov::op::v1::Multiply>(Xi, Xi);
    auto body = make_shared<Model>(OutputVector{Zo, body_condition}, ParameterVector{Xi, current_iteration});

    auto loop = make_shared<ov::op::v5::Loop>(T, exec_condition);
    loop->set_function(body);
    loop->set_special_body_ports(ov::op::v5::Loop::SpecialBodyPorts{1, 1});

    loop->set_sliced_input(Xi, X, 0, 1, 1, -1, 0);

    // check input descriptors
    for (auto& desc : loop->get_input_descriptions()) {
        auto type_info = desc->get_type_info();
        if (std::strcmp(type_info.name, "InvariantInputDescription") == 0) {
            auto input_desc = ov::as_type_ptr<ov::op::v0::TensorIterator::InvariantInputDescription>(desc);
            EXPECT_NE(input_desc, nullptr);
        } else if (std::strcmp(type_info.name, "SliceInputDescription") == 0) {
            auto input_desc = ov::as_type_ptr<ov::op::v0::TensorIterator::SliceInputDescription>(desc);
            EXPECT_NE(input_desc, nullptr);
        } else if (std::strcmp(type_info.name, "MergedInputDescription") == 0) {
            auto input_desc = ov::as_type_ptr<ov::op::v0::TensorIterator::MergedInputDescription>(desc);
            EXPECT_NE(input_desc, nullptr);
        }
    }

    // Output 0 is last Zo
    auto out0 = loop->get_iter_value(body_condition, -1);
    auto out1 = loop->get_iter_value(Zo, -1);
    // Output 1 is concat of Zos
    // start=0, stride=1, part_size=1, end=-1, axis=1
    auto out2 = loop->get_concatenated_slices(Zo, 0, 1, 1, -1, 0);

    // check output descriptors
    for (auto& desc : loop->get_output_descriptions()) {
        auto type_info = desc->get_type_info();
        if (std::strcmp(type_info.name, "ConcatOutputDescription") == 0) {
            auto output_desc = ov::as_type_ptr<ov::op::v0::TensorIterator::ConcatOutputDescription>(desc);
            EXPECT_NE(output_desc, nullptr);
        } else if (std::strcmp(type_info.name, "BodyOutputDescription") == 0) {
            auto output_desc = ov::as_type_ptr<ov::op::v0::TensorIterator::BodyOutputDescription>(desc);
            EXPECT_NE(output_desc, nullptr);
        }
    }
    auto result0 = make_shared<ov::op::v0::Result>(out0);
    auto result1 = make_shared<ov::op::v0::Result>(out1);
    auto result2 = make_shared<ov::op::v0::Result>(out2);
    Shape out0_shape{};
    PartialShape out1_shape{1, Dimension::dynamic(), 10};
    PartialShape out2_shape{Dimension::dynamic(), Dimension::dynamic(), 10};

    auto results = ResultVector{result0, result1, result2};
    auto f = make_shared<Model>(results, ParameterVector{X, T});
    EXPECT_EQ(f->get_output_size(), 3);
    EXPECT_EQ(result0->get_output_shape(0), out0_shape);
    EXPECT_EQ(result1->get_output_partial_shape(0), out1_shape);
    EXPECT_EQ(result2->get_output_partial_shape(0), out2_shape);

    const auto inp0_shape = PartialShape{1, Dimension::dynamic(), 10};
    const auto inp1_shape = Shape{};
    EXPECT_EQ(body->get_parameters().size(), 2);
    EXPECT_EQ(body->get_parameters().at(0)->get_partial_shape(), inp0_shape);
    EXPECT_EQ(body->get_parameters().at(1)->get_shape(), inp1_shape);

    EXPECT_EQ(loop->get_output_size(), 3);
    EXPECT_EQ(loop->get_output_shape(0), out0_shape);
    EXPECT_EQ(loop->get_output_partial_shape(1), out1_shape);
    EXPECT_EQ(loop->get_output_partial_shape(2), out2_shape);
}

// dynamic output
// trip_count = dynamic
// execution_condition = true
// body_condition = true
// input is static shape, sub-model output shapes has one dynamic dimension and this output is a backedge to a
// parameter, other shapes are static
TEST(type_prop, loop_operation_dynamic_iter_static_shapes_inputs_dynamic_shape_outputs) {
    // That which we iterate over
    auto X = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{1, 1, 10});
    auto T = make_shared<ov::op::v0::Parameter>(element::i64, Shape{});

    // Set up the cell body, a function from (Xi) -> Concat(Xi, C) -> (Zo)
    // Body parameters
    auto Xi = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{1, Dimension::dynamic(), 10});

    auto body_condition = make_shared<ov::op::v0::Constant>(element::boolean, Shape{}, true);
    auto trip_count = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, 10);
    auto exec_condition = make_shared<ov::op::v0::Constant>(element::boolean, Shape{}, true);

    // Body
    auto C = ov::op::v0::Constant::create(element::f32, {1, 1, 10}, {0});
    auto Zo = make_shared<ov::op::v0::Concat>(NodeVector{Xi, C}, 1);
    auto Z = make_shared<ov::op::v0::Result>(Zo);
    auto body = make_shared<Model>(OutputVector{Z, body_condition}, ParameterVector{Xi});

    auto loop = make_shared<ov::op::v5::Loop>(T, exec_condition);
    loop->set_function(body);
    loop->set_special_body_ports(ov::op::v5::Loop::SpecialBodyPorts{-1, 1});
    loop->set_merged_input(Xi, X, Z);

    // check input descriptors
    for (auto& desc : loop->get_input_descriptions()) {
        auto type_info = desc->get_type_info();
        if (std::strcmp(type_info.name, "InvariantInputDescription") == 0) {
            auto input_desc = ov::as_type_ptr<ov::op::v0::TensorIterator::InvariantInputDescription>(desc);
            EXPECT_NE(input_desc, nullptr);
        } else if (std::strcmp(type_info.name, "SliceInputDescription") == 0) {
            auto input_desc = ov::as_type_ptr<ov::op::v0::TensorIterator::SliceInputDescription>(desc);
            EXPECT_NE(input_desc, nullptr);
        } else if (std::strcmp(type_info.name, "MergedInputDescription") == 0) {
            auto input_desc = ov::as_type_ptr<ov::op::v0::TensorIterator::MergedInputDescription>(desc);
            EXPECT_NE(input_desc, nullptr);
        }
    }

    // Output 1 is last Z
    auto out0 = loop->get_iter_value(body_condition, -1);
    auto out1 = loop->get_iter_value(Z, -1);

    // check output descriptors
    for (auto& desc : loop->get_output_descriptions()) {
        auto type_info = desc->get_type_info();
        if (std::strcmp(type_info.name, "ConcatOutputDescription") == 0) {
            auto output_desc = ov::as_type_ptr<ov::op::v0::TensorIterator::ConcatOutputDescription>(desc);
            EXPECT_NE(output_desc, nullptr);
        } else if (std::strcmp(type_info.name, "BodyOutputDescription") == 0) {
            auto output_desc = ov::as_type_ptr<ov::op::v0::TensorIterator::BodyOutputDescription>(desc);
            EXPECT_NE(output_desc, nullptr);
        }
    }
    auto result0 = make_shared<ov::op::v0::Result>(out0);
    auto result1 = make_shared<ov::op::v0::Result>(out1);
    Shape out0_shape{};
    PartialShape out1_shape{1, Dimension::dynamic(), 10};

    auto results = ResultVector{result0, result1};
    auto f = make_shared<Model>(results, ParameterVector{X, T});
    EXPECT_EQ(f->get_output_size(), 2);
    EXPECT_EQ(result0->get_output_shape(0), out0_shape);
    // should be dynamic
    EXPECT_TRUE(result1->get_output_partial_shape(0).compatible(out1_shape));

    const auto inp0_shape = PartialShape{1, Dimension::dynamic(), 10};
    const auto inp1_shape = Shape{};
    EXPECT_EQ(body->get_parameters().size(), 1);
    // backedge, should be also dynamic
    EXPECT_EQ(body->get_parameters().at(0)->get_partial_shape(), inp0_shape);

    EXPECT_EQ(loop->get_output_size(), 2);
    EXPECT_EQ(loop->get_output_shape(0), out0_shape);
    // map from the submodel, should be dynamic
    EXPECT_TRUE(loop->get_output_partial_shape(1).compatible(out1_shape));
}

// dynamic output
// trip_count = dynamic
// execution_condition = true
// body_condition = true
// input is dynamic shape, sub-model output shapes has one dynamic dimension and this output is a backedge to a
// parameter, one dynamic shape and one static shape
TEST(type_prop, loop_operation_dynamic_iter_dynamic_shapes_inputs_dynamic_shape_outputs) {
    // That which we iterate over
    auto X = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, 1, 10});
    auto T = make_shared<ov::op::v0::Parameter>(element::i64, Shape{});

    // Set up the cell body, a function from (Xi) -> Concat(Xi, Xi, 1) -> (Zo)
    // Body parameters
    auto Xi = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, 10});

    auto body_condition = make_shared<ov::op::v0::Constant>(element::boolean, Shape{}, true);
    auto trip_count = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, 10);
    auto exec_condition = make_shared<ov::op::v0::Constant>(element::boolean, Shape{}, true);

    // Body
    auto Zo = make_shared<ov::op::v0::Concat>(NodeVector{Xi, Xi}, 1);
    auto Z = make_shared<ov::op::v0::Result>(Zo);
    auto body = make_shared<Model>(OutputVector{Z, body_condition}, ParameterVector{Xi});

    auto loop = make_shared<ov::op::v5::Loop>(T, exec_condition);
    loop->set_function(body);
    loop->set_special_body_ports(ov::op::v5::Loop::SpecialBodyPorts{-1, 1});
    loop->set_merged_input(Xi, X, Z);

    // check input descriptors
    for (auto& desc : loop->get_input_descriptions()) {
        auto type_info = desc->get_type_info();
        if (std::strcmp(type_info.name, "InvariantInputDescription") == 0) {
            auto input_desc = ov::as_type_ptr<ov::op::v0::TensorIterator::InvariantInputDescription>(desc);
            EXPECT_NE(input_desc, nullptr);
        } else if (std::strcmp(type_info.name, "SliceInputDescription") == 0) {
            auto input_desc = ov::as_type_ptr<ov::op::v0::TensorIterator::SliceInputDescription>(desc);
            EXPECT_NE(input_desc, nullptr);
        } else if (std::strcmp(type_info.name, "MergedInputDescription") == 0) {
            auto input_desc = ov::as_type_ptr<ov::op::v0::TensorIterator::MergedInputDescription>(desc);
            EXPECT_NE(input_desc, nullptr);
        }
    }

    // Output 1 is last Z
    auto out0 = loop->get_iter_value(body_condition, -1);
    auto out1 = loop->get_iter_value(Z, -1);

    // check output descriptors
    for (auto& desc : loop->get_output_descriptions()) {
        auto type_info = desc->get_type_info();
        if (std::strcmp(type_info.name, "ConcatOutputDescription") == 0) {
            auto output_desc = ov::as_type_ptr<ov::op::v0::TensorIterator::ConcatOutputDescription>(desc);
            EXPECT_NE(output_desc, nullptr);
        } else if (std::strcmp(type_info.name, "BodyOutputDescription") == 0) {
            auto output_desc = ov::as_type_ptr<ov::op::v0::TensorIterator::BodyOutputDescription>(desc);
            EXPECT_NE(output_desc, nullptr);
        }
    }
    auto result0 = make_shared<ov::op::v0::Result>(out0);
    auto result1 = make_shared<ov::op::v0::Result>(out1);
    Shape out0_shape{};
    PartialShape out1_shape{-1, -1, 10};

    auto results = ResultVector{result0, result1};
    auto f = make_shared<Model>(results, ParameterVector{X, T});
    EXPECT_EQ(f->get_output_size(), 2);
    EXPECT_EQ(result0->get_output_shape(0), out0_shape);
    // should be dynamic
    EXPECT_EQ(result1->get_output_partial_shape(0), out1_shape);

    const auto inp0_shape = PartialShape{-1, -1, 10};
    const auto inp1_shape = Shape{};
    EXPECT_EQ(body->get_parameters().size(), 1);
    // backedge, should be also dynamic
    EXPECT_EQ(body->get_parameters().at(0)->get_partial_shape(), inp0_shape);

    EXPECT_EQ(loop->get_output_size(), 2);
    EXPECT_EQ(loop->get_output_shape(0), out0_shape);
    // map from the submodel, should be dynamic
    EXPECT_EQ(loop->get_output_partial_shape(1), out1_shape);
}

// dynamic output
// trip_count = dynamic
// execution_condition = true
// body_condition = true
// 2 inputs is dynamic shape, sub-model's 2 output shapes has one dynamic dimension and this output is a backedge to a
// parameter, other shapes are static
// main model:
// Parameter(-1,1,10) Parameter(-1,1,10)   Const/Condition()...
//    |                     |                |
//    |_________Loop________|________________|
//               |
//  ________________________________
//  |     |           |            |
//  r0() r1(-1,-1,10) r2(-1,-1,10) r3(-1,-1,10)
//
// sub model:
//      Parameter1 (-1,-1,10)         Parameter2 (-1,-1,10) Const/Condition
//      |                  |           |                  |     |
//      |_Concat(-1,-1,10)_|           |_Concat(-1,-1,10)_|     |
//         |              |                      |              |
//      Result(-1,-1,10) Result(-1,-1,10)  Result(-1,-1,10)   Result()
//                         |                     |
//                  backedge to Parameter1  backedge to Parameter2
TEST(type_prop, loop_operation_dynamic_iter_dynamic_shapes2_inputs_dynamic_shape_outputs3) {
    // That which we iterate over
    auto X0 = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, 1, 10});
    auto X1 = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, 1, 10});
    auto T = make_shared<ov::op::v0::Parameter>(element::i64, Shape{});

    // Set up the cell body, a function from (Xi0) -> Concat(Xi0, Xi0, 1) -> (Zo0)
    //                                       (Xi1) -> Concat(Xi1, Xi1, 1) -> (Zo1)
    // Body parameters
    auto Xi0 = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, 1, 10});
    auto Xi1 = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, 1, 10});

    auto body_condition = make_shared<ov::op::v0::Constant>(element::boolean, Shape{}, true);
    auto trip_count = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, 10);
    auto exec_condition = make_shared<ov::op::v0::Constant>(element::boolean, Shape{}, true);

    // Body
    auto Zo0 = make_shared<ov::op::v0::Concat>(NodeVector{Xi0, Xi0}, 1);
    auto Zo1 = make_shared<ov::op::v0::Concat>(NodeVector{Xi1, Xi1}, 1);
    auto Y = make_shared<ov::op::v0::Result>(Zo0);
    auto Z0 = make_shared<ov::op::v0::Result>(Zo0);
    auto Z1 = make_shared<ov::op::v0::Result>(Zo1);
    auto body = make_shared<Model>(OutputVector{Y, Z0, Z1, body_condition}, ParameterVector{Xi0, Xi1});

    auto loop = make_shared<ov::op::v5::Loop>(T, exec_condition);
    loop->set_function(body);
    loop->set_special_body_ports(ov::op::v5::Loop::SpecialBodyPorts{-1, 3});
    loop->set_merged_input(Xi0, X0, Z0);
    loop->set_merged_input(Xi1, X1, Z1);

    // check input descriptors
    for (auto& desc : loop->get_input_descriptions()) {
        auto type_info = desc->get_type_info();
        if (std::strcmp(type_info.name, "InvariantInputDescription") == 0) {
            auto input_desc = ov::as_type_ptr<ov::op::v0::TensorIterator::InvariantInputDescription>(desc);
            EXPECT_NE(input_desc, nullptr);
        } else if (std::strcmp(type_info.name, "SliceInputDescription") == 0) {
            auto input_desc = ov::as_type_ptr<ov::op::v0::TensorIterator::SliceInputDescription>(desc);
            EXPECT_NE(input_desc, nullptr);
        } else if (std::strcmp(type_info.name, "MergedInputDescription") == 0) {
            auto input_desc = ov::as_type_ptr<ov::op::v0::TensorIterator::MergedInputDescription>(desc);
            EXPECT_NE(input_desc, nullptr);
        }
    }

    // Output 1 is last Z
    auto out0 = loop->get_iter_value(body_condition, -1);
    auto out1 = loop->get_iter_value(Y, -1);
    auto out2 = loop->get_iter_value(Z0, -1);
    auto out3 = loop->get_iter_value(Z1, -1);

    // check output descriptors
    for (auto& desc : loop->get_output_descriptions()) {
        auto type_info = desc->get_type_info();
        if (std::strcmp(type_info.name, "ConcatOutputDescription") == 0) {
            auto output_desc = ov::as_type_ptr<ov::op::v0::TensorIterator::ConcatOutputDescription>(desc);
            EXPECT_NE(output_desc, nullptr);
        } else if (std::strcmp(type_info.name, "BodyOutputDescription") == 0) {
            auto output_desc = ov::as_type_ptr<ov::op::v0::TensorIterator::BodyOutputDescription>(desc);
            EXPECT_NE(output_desc, nullptr);
        }
    }
    auto result0 = make_shared<ov::op::v0::Result>(out0);
    auto result1 = make_shared<ov::op::v0::Result>(out1);
    auto result2 = make_shared<ov::op::v0::Result>(out2);
    auto result3 = make_shared<ov::op::v0::Result>(out3);
    Shape out0_shape{};
    PartialShape out1_shape{-1, -1, 10};

    auto results = ResultVector{result0, result1, result2, result3};
    auto f = make_shared<Model>(results, ParameterVector{X0, X1, T});
    EXPECT_EQ(f->get_output_size(), 4);
    EXPECT_EQ(result0->get_output_shape(0), out0_shape);
    // should be dynamic
    EXPECT_EQ(result1->get_output_partial_shape(0), out1_shape);
    EXPECT_EQ(result2->get_output_partial_shape(0), out1_shape);
    EXPECT_EQ(result3->get_output_partial_shape(0), out1_shape);

    const auto inp0_shape = PartialShape{-1, -1, 10};
    const auto inp1_shape = Shape{};
    EXPECT_EQ(body->get_parameters().size(), 2);
    // backedge, should be also dynamic
    EXPECT_EQ(body->get_parameters().at(0)->get_partial_shape(), inp0_shape);
    EXPECT_EQ(body->get_parameters().at(1)->get_partial_shape(), inp0_shape);

    EXPECT_EQ(loop->get_output_size(), 4);
    EXPECT_EQ(loop->get_output_shape(0), out0_shape);
    // map from the submodel, should be dynamic
    EXPECT_EQ(loop->get_output_partial_shape(1), out1_shape);
    EXPECT_EQ(loop->get_output_partial_shape(2), out1_shape);
    EXPECT_EQ(loop->get_output_partial_shape(3), out1_shape);
}

// dynamic output
// trip_count = dynamic
// execution_condition = true
// body_condition = true
// input is 1D shape, sub-model output shapes has one dynamic dimension and this output is a backedge to a
// parameter, one dynamic shape and one static shape
TEST(type_prop, loop_operation_dynamic_iter_1d_shapes_inputs_dynamic_shape_outputs) {
    // That which we iterate over
    auto X = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{1});
    auto T = make_shared<ov::op::v0::Parameter>(element::i64, Shape{});

    // Set up the cell body, a function from (Xi) -> Concat(Xi, Xi, 1) -> (Zo)
    // Body parameters
    auto Xi = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{1});

    auto body_condition = make_shared<ov::op::v0::Constant>(element::boolean, Shape{}, true);
    auto trip_count = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, 10);
    auto exec_condition = make_shared<ov::op::v0::Constant>(element::boolean, Shape{}, true);

    // Body
    auto X0 = make_shared<ov::op::v1::Reshape>(Xi, ov::op::v0::Constant::create(ov::element::i32, {1}, {-1}), false);
    auto Zo = make_shared<ov::op::v0::Concat>(NodeVector{X0, X0}, 0);
    auto Z = make_shared<ov::op::v0::Result>(Zo);
    auto body = make_shared<Model>(OutputVector{Z, body_condition}, ParameterVector{Xi});

    auto loop = make_shared<ov::op::v5::Loop>(T, exec_condition);
    loop->set_function(body);
    loop->set_special_body_ports(ov::op::v5::Loop::SpecialBodyPorts{-1, 1});
    loop->set_merged_input(Xi, X, Z);

    // check input descriptors
    for (auto& desc : loop->get_input_descriptions()) {
        auto type_info = desc->get_type_info();
        if (std::strcmp(type_info.name, "InvariantInputDescription") == 0) {
            auto input_desc = ov::as_type_ptr<ov::op::v0::TensorIterator::InvariantInputDescription>(desc);
            EXPECT_NE(input_desc, nullptr);
        } else if (std::strcmp(type_info.name, "SliceInputDescription") == 0) {
            auto input_desc = ov::as_type_ptr<ov::op::v0::TensorIterator::SliceInputDescription>(desc);
            EXPECT_NE(input_desc, nullptr);
        } else if (std::strcmp(type_info.name, "MergedInputDescription") == 0) {
            auto input_desc = ov::as_type_ptr<ov::op::v0::TensorIterator::MergedInputDescription>(desc);
            EXPECT_NE(input_desc, nullptr);
        }
    }

    // Output 1 is last Z
    auto out0 = loop->get_iter_value(body_condition, -1);
    auto out1 = loop->get_iter_value(Z, -1);

    // check output descriptors
    for (auto& desc : loop->get_output_descriptions()) {
        auto type_info = desc->get_type_info();
        if (std::strcmp(type_info.name, "ConcatOutputDescription") == 0) {
            auto output_desc = ov::as_type_ptr<ov::op::v0::TensorIterator::ConcatOutputDescription>(desc);
            EXPECT_NE(output_desc, nullptr);
        } else if (std::strcmp(type_info.name, "BodyOutputDescription") == 0) {
            auto output_desc = ov::as_type_ptr<ov::op::v0::TensorIterator::BodyOutputDescription>(desc);
            EXPECT_NE(output_desc, nullptr);
        }
    }
    auto result0 = make_shared<ov::op::v0::Result>(out0);
    auto result1 = make_shared<ov::op::v0::Result>(out1);
    Shape out0_shape{};
    PartialShape out1_shape{-1};

    auto results = ResultVector{result0, result1};
    auto f = make_shared<Model>(results, ParameterVector{X, T});
    EXPECT_EQ(f->get_output_size(), 2);
    EXPECT_EQ(result0->get_output_shape(0), out0_shape);
    // should be dynamic
    EXPECT_EQ(result1->get_output_partial_shape(0), out1_shape);

    const auto inp0_shape = PartialShape{-1};
    const auto inp1_shape = Shape{};
    EXPECT_EQ(body->get_parameters().size(), 1);
    // backedge, should be also dynamic
    EXPECT_EQ(body->get_parameters().at(0)->get_partial_shape(), inp0_shape);

    EXPECT_EQ(loop->get_output_size(), 2);
    EXPECT_EQ(loop->get_output_shape(0), out0_shape);
    // map from the submodel, should be dynamic
    EXPECT_EQ(loop->get_output_partial_shape(1), out1_shape);
}

// dynamic output
// trip_count = -1
// execution_condition = true
// body_condition = true
// model could be described like so:
// Parameter([-1, -1])
// while (true) {
//    input = unsqueeze(input, 0);
// }
TEST(type_prop, loop_operation_dynamic_iter_dynamic_shapes_unsqueeze) {
    // Inner model
    const auto inner_parameter = std::make_shared<ov::op::v0::Parameter>(element::dynamic, ov::PartialShape::dynamic());
    const auto unsqueeze =
        std::make_shared<ov::op::v0::Unsqueeze>(inner_parameter, ov::op::v0::Constant::create(element::i64, {1}, {0}));
    const auto true_const = ov::op::v0::Constant::create(element::boolean, {1}, {1});
    auto body = std::make_shared<Model>(OutputVector{unsqueeze, true_const}, ParameterVector{inner_parameter});

    // Outer model
    const auto outer_parameter =
        std::make_shared<ov::op::v0::Parameter>(element::dynamic, ov::PartialShape::dynamic(2));

    const auto trip_count = ov::op::v0::Constant::create(element::i64, {1}, {-1});
    const auto execution_condition = ov::op::v0::Constant::create(element::boolean, {1}, {1});
    const auto loop = std::make_shared<ov::op::v5::Loop>(trip_count, execution_condition);
    loop->set_function(body);
    loop->set_merged_input(inner_parameter, outer_parameter, unsqueeze);
    loop->set_special_body_ports(ov::op::v5::Loop::SpecialBodyPorts{-1, 1});

    auto outer_result = make_shared<ov::op::v0::Result>(loop->get_iter_value(unsqueeze, -1));

    auto outer_model = std::make_shared<Model>(ResultVector{outer_result}, ParameterVector{outer_parameter});
    PartialShape outer_shape = PartialShape::dynamic();
    EXPECT_EQ(outer_model->get_output_size(), 1);
    EXPECT_EQ(outer_result->get_output_partial_shape(0), outer_shape);
}

// A merged (back-edged) input is seeded with a statically-empty dimension
// ([0, 4]) while the body grows that axis on every iteration by concatenating
// the carried tensor with a [1, 4] value. The grown tensor is loop-carried
// STATE only - it feeds the back-edge but is not exposed as a Loop output (only
// a reduction of it is). The shape-merge driven by the output descriptions never
// visits this back-edge, so the merged body Parameter's static-0 dimension - which
// describes only the outer initial value, not the loop-carried shape - must be
// relaxed to dynamic by the dedicated merged-input pass. Without it the Parameter
// stays clamped at 0, freezing the Concat at an empty tensor.
TEST(type_prop, loop_merged_state_only_input_zero_dim_widened_by_growing_body) {
    // Body: grown = concat(carried, [1, 4]); the loop output is a per-row reduction of grown.
    auto body_carried = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{0, 4});
    auto step = ov::op::v0::Constant::create(element::f32, Shape{1, 4}, std::vector<float>(4, 1.f));
    auto grown = make_shared<ov::op::v0::Concat>(OutputVector{body_carried, step}, 0);
    auto reduce_axis = ov::op::v0::Constant::create(element::i64, Shape{1}, {1});
    auto reduced = make_shared<ov::op::v1::ReduceSum>(grown, reduce_axis);  // [?, 4] -> [?]
    auto body_condition = ov::op::v0::Constant::create(element::boolean, Shape{1}, {true});
    auto body = make_shared<Model>(OutputVector{body_condition, grown, reduced}, ParameterVector{body_carried});

    auto init = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{0, 4});
    auto trip_count = ov::op::v0::Constant::create(element::i64, Shape{1}, {5});
    auto exec_condition = ov::op::v0::Constant::create(element::boolean, Shape{1}, {true});

    auto loop = make_shared<ov::op::v5::Loop>(trip_count, exec_condition);
    loop->set_function(body);
    loop->set_special_body_ports(ov::op::v5::Loop::SpecialBodyPorts{-1, 0});
    // grown is the back-edge (state); only `reduced` is exposed as a Loop output.
    loop->set_merged_input(body_carried, init, grown);
    auto out = loop->get_iter_value(reduced, -1);
    auto result = make_shared<ov::op::v0::Result>(out);
    auto f = make_shared<Model>(ResultVector{result}, ParameterVector{init});

    // The state-only merged input's static-0 axis is relaxed to dynamic (not clamped at 0),
    // so the body Concat and its reduction are typed over a growable, non-empty tensor.
    EXPECT_EQ(body_carried->get_partial_shape(), (PartialShape{Dimension::dynamic(), 4}));
    EXPECT_EQ(loop->get_output_partial_shape(0), PartialShape{Dimension(1, -1)});
}

// Same pattern as loop_merged_state_only_input_zero_dim_widened_by_growing_body
// but the per-iteration growth tensor has a *dynamically*-known row count ([?, 4]
// instead of [1, 4]).  The body Concat therefore also produces [?, 4], so the
// back-edge reconciliation finds it *compatible* with the frozen [0, 4] parameter
// (Dimension::dynamic().compatible(Dimension(0)) == true) and never widens it on
// its own.  Only the up-front zero-dim widening pass prevents body_carried from
// being clamped at 0 for the entire shape inference.
TEST(type_prop, loop_merged_state_only_input_zero_dim_widened_dynamic_step) {
    // Body: grown = concat(carried, step); step row-count is unknown at compile time.
    auto body_carried = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{0, 4});
    auto body_step = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 4});
    auto grown = make_shared<ov::op::v0::Concat>(OutputVector{body_carried, body_step}, 0);
    auto reduce_axis = ov::op::v0::Constant::create(element::i64, Shape{1}, {1});
    auto reduced = make_shared<ov::op::v1::ReduceSum>(grown, reduce_axis);
    auto body_condition = ov::op::v0::Constant::create(element::boolean, Shape{1}, {true});
    auto body =
        make_shared<Model>(OutputVector{body_condition, grown, reduced}, ParameterVector{body_carried, body_step});

    auto init = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{0, 4});
    auto outer_step = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 4});
    auto trip_count = ov::op::v0::Constant::create(element::i64, Shape{1}, {5});
    auto exec_condition = ov::op::v0::Constant::create(element::boolean, Shape{1}, {true});

    auto loop = make_shared<ov::op::v5::Loop>(trip_count, exec_condition);
    loop->set_function(body);
    loop->set_special_body_ports(ov::op::v5::Loop::SpecialBodyPorts{-1, 0});
    loop->set_merged_input(body_carried, init, grown);
    loop->set_invariant_input(body_step, outer_step);
    auto out = loop->get_iter_value(reduced, -1);
    auto result = make_shared<ov::op::v0::Result>(out);
    auto f = make_shared<Model>(ResultVector{result}, ParameterVector{init, outer_step});

    // Concat({0,4}, {?,4}) yields {?,4}; dynamic.compatible(0) is true, so the
    // back-edge reconciliation loop alone cannot widen body_carried away from 0.
    // The zero-dim widening pass must relax it to dynamic before reconciliation runs.
    EXPECT_EQ(body_carried->get_partial_shape(), (PartialShape{Dimension::dynamic(), 4}));
    EXPECT_EQ(loop->get_output_partial_shape(0), PartialShape{Dimension::dynamic()});
}
