//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/opsets/opset5.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

// trip_count = 10
// execution_condition = true
// body_condition = true
// all shapes are static, 10 iterations will be executed
TEST(type_prop, loop_operation_for_mode_10_iter_static_shapes)
{
    // That which we iterate over
    auto X = make_shared<opset5::Parameter>(element::f32, Shape{32, 1, 10});
    auto Y = make_shared<opset5::Parameter>(element::f32, Shape{32, 1, 10});
    auto M = make_shared<opset5::Parameter>(element::f32, Shape{32, 1, 10});

    // Set up the cell body, a function from (Xi, Yi) -> (Zo)
    // Body parameters
    auto current_iteration = make_shared<opset5::Parameter>(element::i64, Shape{1});
    auto Xi = make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());
    auto Yi = make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());
    auto M_body = make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());
    auto body_condition = std::make_shared<ngraph::opset5::Constant>(
        ngraph::element::boolean, ngraph::Shape{1}, true);

    auto trip_count =
        std::make_shared<ngraph::opset5::Constant>(ngraph::element::i64, ngraph::Shape{1}, 10);
    auto exec_condition = std::make_shared<ngraph::opset5::Constant>(
        ngraph::element::boolean, ngraph::Shape{1}, true);
    // Body
    auto sum = make_shared<ngraph::opset5::Add>(Xi, Yi);
    auto Zo = make_shared<ngraph::opset5::Multiply>(sum, M_body);
    auto body = make_shared<ngraph::Function>(OutputVector{body_condition, Zo},
                                              ParameterVector{current_iteration, Xi, Yi, M_body});

    auto loop = make_shared<opset5::Loop>(trip_count, exec_condition);
    loop->set_function(body);

    loop->set_invariant_input(Xi, X);
    loop->set_invariant_input(Yi, Y);
    loop->set_merged_input(M_body, M, Zo);

    loop->set_special_body_ports(ngraph::opset5::Loop::SpecialBodyPorts{-1, 0});

    // check input descriptors
    for (auto& desc : loop->get_input_descriptions())
    {
        auto type_info = desc->get_type_info();
        if (std::strcmp(type_info.name, "InvariantInputDescription") == 0)
        {
            auto input_desc =
                as_type_ptr<ngraph::opset5::TensorIterator::InvariantInputDescription>(desc);
            EXPECT_NE(input_desc, nullptr);
        }
        else if (std::strcmp(type_info.name, "SliceInputDescription") == 0)
        {
            auto input_desc =
                as_type_ptr<ngraph::opset5::TensorIterator::SliceInputDescription>(desc);
            EXPECT_NE(input_desc, nullptr);
        }
        else if (std::strcmp(type_info.name, "MergedInputDescription") == 0)
        {
            auto input_desc =
                as_type_ptr<ngraph::opset5::TensorIterator::MergedInputDescription>(desc);
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
    for (auto& desc : loop->get_output_descriptions())
    {
        auto type_info = desc->get_type_info();
        if (std::strcmp(type_info.name, "ConcatOutputDescription") == 0)
        {
            auto output_desc =
                as_type_ptr<ngraph::opset5::TensorIterator::ConcatOutputDescription>(desc);
            EXPECT_NE(output_desc, nullptr);
        }
        else if (std::strcmp(type_info.name, "BodyOutputDescription") == 0)
        {
            auto output_desc =
                as_type_ptr<ngraph::opset5::TensorIterator::BodyOutputDescription>(desc);
            EXPECT_NE(output_desc, nullptr);
        }
    }
    auto result0 = make_shared<opset5::Result>(out0);
    auto result1 = make_shared<opset5::Result>(out1);
    auto result2 = make_shared<opset5::Result>(out2);
    Shape out0_shape{1};
    Shape out1_shape{32, 1, 10};
    Shape out2_shape{32, 10, 10};

    auto results = ResultVector{result0, result1, result2};
    auto f = make_shared<Function>(results, ParameterVector{X, Y, M});
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
TEST(type_prop, loop_operation_dowhile_mode_1_iter_static_shapes)
{
    // That which we iterate over
    auto X = make_shared<opset5::Parameter>(element::f32, Shape{32, 1, 10});
    auto Y = make_shared<opset5::Parameter>(element::f32, Shape{32, 1, 10});
    auto M = make_shared<opset5::Parameter>(element::f32, Shape{32, 1, 10});

    // Set up the cell body, a function from (Xi, Yi) -> (Zo)
    // Body parameters
    auto current_iteration = make_shared<opset5::Parameter>(element::i64, Shape{1});
    auto Xi = make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());
    auto Yi = make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());
    auto M_body = make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());
    auto body_condition = std::make_shared<ngraph::opset5::Constant>(
        ngraph::element::boolean, ngraph::Shape{1}, false);

    auto trip_count =
        std::make_shared<ngraph::opset5::Constant>(ngraph::element::i64, ngraph::Shape{1}, 10);
    auto exec_condition = std::make_shared<ngraph::opset5::Constant>(
        ngraph::element::boolean, ngraph::Shape{1}, true);
    // Body
    auto sum = make_shared<ngraph::opset5::Add>(Xi, Yi);
    auto Zo = make_shared<ngraph::opset5::Multiply>(sum, M_body);
    auto body = make_shared<ngraph::Function>(OutputVector{body_condition, Zo},
                                              ParameterVector{current_iteration, Xi, Yi, M_body});

    auto loop = make_shared<opset5::Loop>(trip_count, exec_condition);
    loop->set_function(body);
    loop->set_special_body_ports(ngraph::opset5::Loop::SpecialBodyPorts{-1, 0});

    loop->set_invariant_input(Xi, X);
    loop->set_invariant_input(Yi, Y);
    loop->set_merged_input(M_body, M, Zo);

    // check input descriptors
    for (auto& desc : loop->get_input_descriptions())
    {
        auto type_info = desc->get_type_info();
        if (std::strcmp(type_info.name, "InvariantInputDescription") == 0)
        {
            auto input_desc =
                as_type_ptr<ngraph::opset5::TensorIterator::InvariantInputDescription>(desc);
            EXPECT_NE(input_desc, nullptr);
        }
        else if (std::strcmp(type_info.name, "SliceInputDescription") == 0)
        {
            auto input_desc =
                as_type_ptr<ngraph::opset5::TensorIterator::SliceInputDescription>(desc);
            EXPECT_NE(input_desc, nullptr);
        }
        else if (std::strcmp(type_info.name, "MergedInputDescription") == 0)
        {
            auto input_desc =
                as_type_ptr<ngraph::opset5::TensorIterator::MergedInputDescription>(desc);
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
    for (auto& desc : loop->get_output_descriptions())
    {
        auto type_info = desc->get_type_info();
        if (std::strcmp(type_info.name, "ConcatOutputDescription") == 0)
        {
            auto output_desc =
                as_type_ptr<ngraph::opset5::TensorIterator::ConcatOutputDescription>(desc);
            EXPECT_NE(output_desc, nullptr);
        }
        else if (std::strcmp(type_info.name, "BodyOutputDescription") == 0)
        {
            auto output_desc =
                as_type_ptr<ngraph::opset5::TensorIterator::BodyOutputDescription>(desc);
            EXPECT_NE(output_desc, nullptr);
        }
    }
    auto result0 = make_shared<opset5::Result>(out0);
    auto result1 = make_shared<opset5::Result>(out1);
    auto result2 = make_shared<opset5::Result>(out2);
    Shape out0_shape{1};
    Shape out1_shape{32, 1, 10};
    Shape out2_shape{32, 1, 10};

    auto results = ResultVector{result0, result1, result2};
    auto f = make_shared<Function>(results, ParameterVector{X, Y, M});
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
TEST(type_prop, loop_operation_for_and_condition_mode_dynamic_iter_static_shapes)
{
    // That which we iterate over
    auto X = make_shared<opset5::Parameter>(element::f32, Shape{1});
    auto Y = make_shared<opset5::Parameter>(element::f32, Shape{1});
    auto M = make_shared<opset5::Parameter>(element::f32, Shape{1});

    // Set up the cell body, a function from (Xi, Yi) -> (Zo)
    // Body parameters
    auto current_iteration = make_shared<opset5::Parameter>(element::i64, Shape{1});
    auto Xi = make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());
    auto Yi = make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());
    auto M_body = make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());
    auto condition_const =
        std::make_shared<ngraph::opset5::Constant>(ngraph::element::f32, ngraph::Shape{1}, 10);
    auto body_condition = std::make_shared<ngraph::opset5::Greater>(M_body, condition_const);

    auto trip_count =
        std::make_shared<ngraph::opset5::Constant>(ngraph::element::i64, ngraph::Shape{1}, 10);
    auto exec_condition = std::make_shared<ngraph::opset5::Constant>(
        ngraph::element::boolean, ngraph::Shape{1}, true);
    // Body
    auto sum = make_shared<ngraph::opset5::Add>(Xi, Yi);
    auto Zo = make_shared<ngraph::opset5::Multiply>(sum, M_body);
    auto body = make_shared<ngraph::Function>(OutputVector{body_condition, Zo},
                                              ParameterVector{Xi, Yi, M_body});

    auto loop = make_shared<opset5::Loop>(trip_count, exec_condition);
    loop->set_function(body);
    loop->set_special_body_ports(ngraph::opset5::Loop::SpecialBodyPorts{-1, 0});

    loop->set_invariant_input(Xi, X);
    loop->set_invariant_input(Yi, Y);
    loop->set_merged_input(M_body, M, Zo);

    // check input descriptors
    for (auto& desc : loop->get_input_descriptions())
    {
        auto type_info = desc->get_type_info();
        if (std::strcmp(type_info.name, "InvariantInputDescription") == 0)
        {
            auto input_desc =
                as_type_ptr<ngraph::opset5::TensorIterator::InvariantInputDescription>(desc);
            EXPECT_NE(input_desc, nullptr);
        }
        else if (std::strcmp(type_info.name, "SliceInputDescription") == 0)
        {
            auto input_desc =
                as_type_ptr<ngraph::opset5::TensorIterator::SliceInputDescription>(desc);
            EXPECT_NE(input_desc, nullptr);
        }
        else if (std::strcmp(type_info.name, "MergedInputDescription") == 0)
        {
            auto input_desc =
                as_type_ptr<ngraph::opset5::TensorIterator::MergedInputDescription>(desc);
            EXPECT_NE(input_desc, nullptr);
        }
    }

    // Output 0 is last Zo
    auto out0 = loop->get_iter_value(body_condition, -1);
    auto out1 = loop->get_iter_value(Zo, -1);

    // check output descriptors
    for (auto& desc : loop->get_output_descriptions())
    {
        auto type_info = desc->get_type_info();
        if (std::strcmp(type_info.name, "ConcatOutputDescription") == 0)
        {
            auto output_desc =
                as_type_ptr<ngraph::opset5::TensorIterator::ConcatOutputDescription>(desc);
            EXPECT_NE(output_desc, nullptr);
        }
        else if (std::strcmp(type_info.name, "BodyOutputDescription") == 0)
        {
            auto output_desc =
                as_type_ptr<ngraph::opset5::TensorIterator::BodyOutputDescription>(desc);
            EXPECT_NE(output_desc, nullptr);
        }
    }
    auto result0 = make_shared<opset5::Result>(out0);
    auto result1 = make_shared<opset5::Result>(out1);
    Shape out0_shape{1};
    Shape out1_shape{1};

    auto results = ResultVector{result0, result1};
    auto f = make_shared<Function>(results, ParameterVector{X, Y, M});
    EXPECT_EQ(result0->get_output_shape(0), out0_shape);
    EXPECT_EQ(result1->get_output_shape(0), out1_shape);

    EXPECT_EQ(loop->get_output_shape(0), out0_shape);
    EXPECT_EQ(loop->get_output_shape(1), out1_shape);
}

// trip_count = 10
// execution_condition = true
// body_condition is not a Constant
// concat output has only dynamic rank, another outputs are static
TEST(type_prop, loop_operation_for_and_condition_mode_dynamic_iter_dynamic_shapes)
{
    // That which we iterate over
    auto X = make_shared<opset5::Parameter>(element::f32, Shape{1});
    auto Y = make_shared<opset5::Parameter>(element::f32, Shape{1});
    auto M = make_shared<opset5::Parameter>(element::f32, Shape{1});

    // Set up the cell body, a function from (Xi, Yi) -> (Zo)
    // Body parameters
    auto current_iteration = make_shared<opset5::Parameter>(element::i64, Shape{1});
    auto Xi = make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());
    auto Yi = make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());
    auto M_body = make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());
    auto condition_const =
        std::make_shared<ngraph::opset5::Constant>(ngraph::element::f32, ngraph::Shape{1}, 10);
    auto body_condition = std::make_shared<ngraph::opset5::Greater>(M_body, condition_const);

    auto trip_count =
        std::make_shared<ngraph::opset5::Constant>(ngraph::element::i64, ngraph::Shape{1}, 10);
    auto exec_condition = std::make_shared<ngraph::opset5::Constant>(
        ngraph::element::boolean, ngraph::Shape{1}, true);
    // Body
    auto sum = make_shared<ngraph::opset5::Add>(Xi, Yi);
    auto Zo = make_shared<ngraph::opset5::Multiply>(sum, M_body);
    auto body = make_shared<ngraph::Function>(OutputVector{body_condition, Zo},
                                              ParameterVector{current_iteration, Xi, Yi, M_body});

    auto loop = make_shared<opset5::Loop>(trip_count, exec_condition);
    loop->set_function(body);
    loop->set_special_body_ports(ngraph::opset5::Loop::SpecialBodyPorts{-1, 0});

    loop->set_invariant_input(Xi, X);
    loop->set_invariant_input(Yi, Y);
    loop->set_merged_input(M_body, M, Zo);

    // check input descriptors
    for (auto& desc : loop->get_input_descriptions())
    {
        auto type_info = desc->get_type_info();
        if (std::strcmp(type_info.name, "InvariantInputDescription") == 0)
        {
            auto input_desc =
                as_type_ptr<ngraph::opset5::TensorIterator::InvariantInputDescription>(desc);
            EXPECT_NE(input_desc, nullptr);
        }
        else if (std::strcmp(type_info.name, "SliceInputDescription") == 0)
        {
            auto input_desc =
                as_type_ptr<ngraph::opset5::TensorIterator::SliceInputDescription>(desc);
            EXPECT_NE(input_desc, nullptr);
        }
        else if (std::strcmp(type_info.name, "MergedInputDescription") == 0)
        {
            auto input_desc =
                as_type_ptr<ngraph::opset5::TensorIterator::MergedInputDescription>(desc);
            EXPECT_NE(input_desc, nullptr);
        }
    }

    // Output 0 is last Zo
    auto out0 = loop->get_iter_value(body_condition, -1);
    auto out1 = loop->get_iter_value(Zo, -1);
    auto out2 = loop->get_concatenated_slices(Zo, 0, 1, 1, -1, 0);

    // check output descriptors
    for (auto& desc : loop->get_output_descriptions())
    {
        auto type_info = desc->get_type_info();
        if (std::strcmp(type_info.name, "ConcatOutputDescription") == 0)
        {
            auto output_desc =
                as_type_ptr<ngraph::opset5::TensorIterator::ConcatOutputDescription>(desc);
            EXPECT_NE(output_desc, nullptr);
        }
        else if (std::strcmp(type_info.name, "BodyOutputDescription") == 0)
        {
            auto output_desc =
                as_type_ptr<ngraph::opset5::TensorIterator::BodyOutputDescription>(desc);
            EXPECT_NE(output_desc, nullptr);
        }
    }
    auto result0 = make_shared<opset5::Result>(out0);
    auto result1 = make_shared<opset5::Result>(out1);
    auto result2 = make_shared<opset5::Result>(out2);
    Shape out0_shape{1};
    Shape out1_shape{1};
    PartialShape out2_shape{PartialShape::dynamic(1)};

    auto results = ResultVector{result0, result1, result2};
    auto f = make_shared<Function>(results, ParameterVector{X, Y, M});
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
// inputs have partially known shape
// concat output has dynamic dimension on axis position, another outputs are static
TEST(type_prop, loop_operation_for_and_condition_mode_dynamic_iter_partially_dynamic_shapes)
{
    // That which we iterate over
    auto X =
        make_shared<opset5::Parameter>(element::f32, PartialShape{1, 2, 3, Dimension::dynamic()});
    auto Y =
        make_shared<opset5::Parameter>(element::f32, PartialShape{1, 2, 3, Dimension::dynamic()});
    auto M = make_shared<opset5::Parameter>(element::f32, Shape{1});

    // Set up the cell body, a function from (Xi, Yi) -> (Zo)
    // Body parameters
    auto current_iteration = make_shared<opset5::Parameter>(element::i64, Shape{1});
    auto Xi = make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());
    auto Yi = make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());
    auto M_body = make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());
    auto condition_const =
        std::make_shared<ngraph::opset5::Constant>(ngraph::element::f32, ngraph::Shape{1}, 10);
    auto body_condition = std::make_shared<ngraph::opset5::Greater>(M_body, condition_const);

    auto trip_count =
        std::make_shared<ngraph::opset5::Constant>(ngraph::element::i64, ngraph::Shape{1}, 10);
    auto exec_condition = std::make_shared<ngraph::opset5::Constant>(
        ngraph::element::boolean, ngraph::Shape{1}, true);
    // Body
    auto sum = make_shared<ngraph::opset5::Add>(Xi, Yi);
    auto Zo = make_shared<ngraph::opset5::Multiply>(sum, M_body);
    auto body = make_shared<ngraph::Function>(OutputVector{body_condition, Zo},
                                              ParameterVector{current_iteration, Xi, Yi, M_body});

    auto loop = make_shared<opset5::Loop>(trip_count, exec_condition);
    loop->set_function(body);
    loop->set_special_body_ports(ngraph::opset5::Loop::SpecialBodyPorts{-1, 0});

    loop->set_invariant_input(Xi, X);
    loop->set_invariant_input(Yi, Y);
    loop->set_merged_input(M_body, M, Zo);

    // check input descriptors
    for (auto& desc : loop->get_input_descriptions())
    {
        auto type_info = desc->get_type_info();
        if (std::strcmp(type_info.name, "InvariantInputDescription") == 0)
        {
            auto input_desc =
                as_type_ptr<ngraph::opset5::TensorIterator::InvariantInputDescription>(desc);
            EXPECT_NE(input_desc, nullptr);
        }
        else if (std::strcmp(type_info.name, "SliceInputDescription") == 0)
        {
            auto input_desc =
                as_type_ptr<ngraph::opset5::TensorIterator::SliceInputDescription>(desc);
            EXPECT_NE(input_desc, nullptr);
        }
        else if (std::strcmp(type_info.name, "MergedInputDescription") == 0)
        {
            auto input_desc =
                as_type_ptr<ngraph::opset5::TensorIterator::MergedInputDescription>(desc);
            EXPECT_NE(input_desc, nullptr);
        }
    }

    // Output 0 is last Zo
    auto out0 = loop->get_iter_value(body_condition, -1);
    auto out1 = loop->get_iter_value(Zo, -1);
    // axis=1 so sliced output on this dimension will be dynamic
    auto out2 = loop->get_concatenated_slices(Zo, 0, 1, 1, -1, 1);

    // check output descriptors
    for (auto& desc : loop->get_output_descriptions())
    {
        auto type_info = desc->get_type_info();
        if (std::strcmp(type_info.name, "ConcatOutputDescription") == 0)
        {
            auto output_desc =
                as_type_ptr<ngraph::opset5::TensorIterator::ConcatOutputDescription>(desc);
            EXPECT_NE(output_desc, nullptr);
        }
        else if (std::strcmp(type_info.name, "BodyOutputDescription") == 0)
        {
            auto output_desc =
                as_type_ptr<ngraph::opset5::TensorIterator::BodyOutputDescription>(desc);
            EXPECT_NE(output_desc, nullptr);
        }
    }
    auto result0 = make_shared<opset5::Result>(out0);
    auto result1 = make_shared<opset5::Result>(out1);
    auto result2 = make_shared<opset5::Result>(out2);
    Shape out0_shape{1};
    PartialShape out1_shape{1, 2, 3, Dimension::dynamic()};
    PartialShape out2_shape{1, Dimension::dynamic(), 3, Dimension::dynamic()};

    auto results = ResultVector{result0, result1, result2};
    auto f = make_shared<Function>(results, ParameterVector{X, Y, M});
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
TEST(type_prop, loop_operation_for_and_condition_mode_dynamic_iter_incorrect_sliced_output_axis)
{
    // That which we iterate over
    auto X =
        make_shared<opset5::Parameter>(element::f32, PartialShape{1, 2, 3, Dimension::dynamic()});
    auto Y =
        make_shared<opset5::Parameter>(element::f32, PartialShape{1, 2, 3, Dimension::dynamic()});
    auto M = make_shared<opset5::Parameter>(element::f32, Shape{1});

    // Set up the cell body, a function from (Xi, Yi) -> (Zo)
    // Body parameters
    auto current_iteration = make_shared<opset5::Parameter>(element::i64, Shape{1});
    auto Xi = make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());
    auto Yi = make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());
    auto M_body = make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());
    auto condition_const =
        std::make_shared<ngraph::opset5::Constant>(ngraph::element::f32, ngraph::Shape{1}, 10);
    auto body_condition = std::make_shared<ngraph::opset5::Greater>(M_body, condition_const);

    auto trip_count =
        std::make_shared<ngraph::opset5::Constant>(ngraph::element::i64, ngraph::Shape{1}, 10);
    auto exec_condition = std::make_shared<ngraph::opset5::Constant>(
        ngraph::element::boolean, ngraph::Shape{1}, true);
    // Body
    auto sum = make_shared<ngraph::opset5::Add>(Xi, Yi);
    auto Zo = make_shared<ngraph::opset5::Multiply>(sum, M_body);
    auto body = make_shared<ngraph::Function>(OutputVector{body_condition, Zo},
                                              ParameterVector{current_iteration, Xi, Yi, M_body});

    auto loop = make_shared<opset5::Loop>(trip_count, exec_condition);
    loop->set_function(body);
    loop->set_special_body_ports(ngraph::opset5::Loop::SpecialBodyPorts{-1, 0});

    loop->set_invariant_input(Xi, X);
    loop->set_invariant_input(Yi, Y);
    loop->set_merged_input(M_body, M, Zo);

    const auto sliced_output_axis = 4;
    try
    {
        auto out = loop->get_concatenated_slices(Zo, 0, 1, 1, -1, sliced_output_axis);
        FAIL() << "Loop was created with incorrect axis of concatenated slices output.";
    }
    catch (const std::exception& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("out of the tensor rank range"));
    }
    catch (...)
    {
        FAIL() << "Construction loop operator failed for unexpected reason.";
    }
}

// trip_count = -1
// execution_condition = true
// body_condition = true
// concat output will be dynamic, another outputs are static
TEST(type_prop, loop_operation_infinite_loop_mode_dynamic_iter_dynamic_shapes)
{
    // That which we iterate over
    auto X = make_shared<opset5::Parameter>(element::f32, Shape{32, 1, 10});
    auto Y = make_shared<opset5::Parameter>(element::f32, Shape{32, 1, 10});
    auto M = make_shared<opset5::Parameter>(element::f32, Shape{32, 1, 10});

    // Set up the cell body, a function from (Xi, Yi) -> (Zo)
    // Body parameters
    auto current_iteration = make_shared<opset5::Parameter>(element::i64, Shape{1});
    auto Xi = make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());
    auto Yi = make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());
    auto M_body = make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());
    auto body_condition = std::make_shared<ngraph::opset5::Constant>(
        ngraph::element::boolean, ngraph::Shape{1}, true);

    auto trip_count =
        std::make_shared<ngraph::opset5::Constant>(ngraph::element::i64, ngraph::Shape{1}, -1);
    auto exec_condition = std::make_shared<ngraph::opset5::Constant>(
        ngraph::element::boolean, ngraph::Shape{1}, true);
    // Body
    auto sum = make_shared<ngraph::opset5::Add>(Xi, Yi);
    auto Zo = make_shared<ngraph::opset5::Multiply>(sum, M_body);
    auto body = make_shared<ngraph::Function>(OutputVector{body_condition, Zo},
                                              ParameterVector{current_iteration, Xi, Yi, M_body});

    auto loop = make_shared<opset5::Loop>(trip_count, exec_condition);
    loop->set_function(body);
    loop->set_special_body_ports(ngraph::opset5::Loop::SpecialBodyPorts{-1, 0});

    loop->set_invariant_input(Xi, X);
    loop->set_invariant_input(Yi, Y);
    loop->set_merged_input(M_body, M, Zo);

    // check input descriptors
    for (auto& desc : loop->get_input_descriptions())
    {
        auto type_info = desc->get_type_info();
        if (std::strcmp(type_info.name, "InvariantInputDescription") == 0)
        {
            auto input_desc =
                as_type_ptr<ngraph::opset5::TensorIterator::InvariantInputDescription>(desc);
            EXPECT_NE(input_desc, nullptr);
        }
        else if (std::strcmp(type_info.name, "SliceInputDescription") == 0)
        {
            auto input_desc =
                as_type_ptr<ngraph::opset5::TensorIterator::SliceInputDescription>(desc);
            EXPECT_NE(input_desc, nullptr);
        }
        else if (std::strcmp(type_info.name, "MergedInputDescription") == 0)
        {
            auto input_desc =
                as_type_ptr<ngraph::opset5::TensorIterator::MergedInputDescription>(desc);
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
    for (auto& desc : loop->get_output_descriptions())
    {
        auto type_info = desc->get_type_info();
        if (std::strcmp(type_info.name, "ConcatOutputDescription") == 0)
        {
            auto output_desc =
                as_type_ptr<ngraph::opset5::TensorIterator::ConcatOutputDescription>(desc);
            EXPECT_NE(output_desc, nullptr);
        }
        else if (std::strcmp(type_info.name, "BodyOutputDescription") == 0)
        {
            auto output_desc =
                as_type_ptr<ngraph::opset5::TensorIterator::BodyOutputDescription>(desc);
            EXPECT_NE(output_desc, nullptr);
        }
    }
    auto result0 = make_shared<opset5::Result>(out0);
    auto result1 = make_shared<opset5::Result>(out1);
    auto result2 = make_shared<opset5::Result>(out2);
    Shape out0_shape{1};
    Shape out1_shape{32, 1, 10};
    PartialShape out2_shape{32, Dimension::dynamic(), 10};

    auto results = ResultVector{result0, result1, result2};
    auto f = make_shared<Function>(results, ParameterVector{X, Y, M});
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
TEST(type_prop, loop_operation_for_mode_10_iter_static_shapes_special_body_ports)
{
    // That which we iterate over
    auto X = make_shared<opset5::Parameter>(element::f32, Shape{32, 1, 10});
    auto Y = make_shared<opset5::Parameter>(element::f32, Shape{32, 1, 10});
    auto M = make_shared<opset5::Parameter>(element::f32, Shape{32, 1, 10});

    // Set up the cell body, a function from (Xi, Yi) -> (Zo)
    // Body parameters
    auto current_iteration = make_shared<opset5::Parameter>(element::i64, Shape{1});
    auto Xi = make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());
    auto Yi = make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());
    auto M_body = make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());
    auto body_condition = std::make_shared<ngraph::opset5::Constant>(
        ngraph::element::boolean, ngraph::Shape{1}, true);

    auto trip_count =
        std::make_shared<ngraph::opset5::Constant>(ngraph::element::i64, ngraph::Shape{1}, 10);
    auto exec_condition = std::make_shared<ngraph::opset5::Constant>(
        ngraph::element::boolean, ngraph::Shape{1}, true);
    // Body
    auto sum = make_shared<ngraph::opset5::Add>(Xi, Yi);
    auto Zo = make_shared<ngraph::opset5::Multiply>(sum, M_body);
    auto body = make_shared<ngraph::Function>(OutputVector{Zo, body_condition},
                                              ParameterVector{Xi, current_iteration, Yi, M_body});

    auto loop = make_shared<opset5::Loop>(trip_count, exec_condition);
    loop->set_function(body);
    loop->set_special_body_ports(ngraph::opset5::Loop::SpecialBodyPorts{1, 1});

    loop->set_invariant_input(Xi, X);
    loop->set_invariant_input(Yi, Y);
    loop->set_merged_input(M_body, M, Zo);

    // check input descriptors
    for (auto& desc : loop->get_input_descriptions())
    {
        auto type_info = desc->get_type_info();
        if (std::strcmp(type_info.name, "InvariantInputDescription") == 0)
        {
            auto input_desc =
                as_type_ptr<ngraph::opset5::TensorIterator::InvariantInputDescription>(desc);
            EXPECT_NE(input_desc, nullptr);
        }
        else if (std::strcmp(type_info.name, "SliceInputDescription") == 0)
        {
            auto input_desc =
                as_type_ptr<ngraph::opset5::TensorIterator::SliceInputDescription>(desc);
            EXPECT_NE(input_desc, nullptr);
        }
        else if (std::strcmp(type_info.name, "MergedInputDescription") == 0)
        {
            auto input_desc =
                as_type_ptr<ngraph::opset5::TensorIterator::MergedInputDescription>(desc);
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
    for (auto& desc : loop->get_output_descriptions())
    {
        auto type_info = desc->get_type_info();
        if (std::strcmp(type_info.name, "ConcatOutputDescription") == 0)
        {
            auto output_desc =
                as_type_ptr<ngraph::opset5::TensorIterator::ConcatOutputDescription>(desc);
            EXPECT_NE(output_desc, nullptr);
        }
        else if (std::strcmp(type_info.name, "BodyOutputDescription") == 0)
        {
            auto output_desc =
                as_type_ptr<ngraph::opset5::TensorIterator::BodyOutputDescription>(desc);
            EXPECT_NE(output_desc, nullptr);
        }
    }
    auto result0 = make_shared<opset5::Result>(out0);
    auto result1 = make_shared<opset5::Result>(out1);
    auto result2 = make_shared<opset5::Result>(out2);
    Shape out0_shape{1};
    Shape out1_shape{32, 1, 10};
    Shape out2_shape{32, 10, 10};

    auto results = ResultVector{result0, result1, result2};
    auto f = make_shared<Function>(results, ParameterVector{X, Y, M});
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
TEST(type_prop, loop_operation_for_mode_10_iter_static_shapes_special_body_ports_scalars)
{
    // That which we iterate over
    auto X = make_shared<opset5::Parameter>(element::f32, Shape{32, 1, 10});
    auto Y = make_shared<opset5::Parameter>(element::f32, Shape{32, 1, 10});
    auto M = make_shared<opset5::Parameter>(element::f32, Shape{32, 1, 10});

    // Set up the cell body, a function from (Xi, Yi) -> (Zo)
    // Body parameters
    auto current_iteration = make_shared<opset5::Parameter>(element::i64, Shape{});
    auto Xi = make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());
    auto Yi = make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());
    auto M_body = make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());
    auto body_condition =
        std::make_shared<ngraph::opset5::Constant>(ngraph::element::boolean, ngraph::Shape{}, true);

    auto trip_count =
        std::make_shared<ngraph::opset5::Constant>(ngraph::element::i64, ngraph::Shape{}, 10);
    auto exec_condition =
        std::make_shared<ngraph::opset5::Constant>(ngraph::element::boolean, ngraph::Shape{}, true);
    // Body
    auto sum = make_shared<ngraph::opset5::Add>(Xi, Yi);
    auto Zo = make_shared<ngraph::opset5::Multiply>(sum, M_body);
    auto body = make_shared<ngraph::Function>(OutputVector{Zo, body_condition},
                                              ParameterVector{Xi, current_iteration, Yi, M_body});

    auto loop = make_shared<opset5::Loop>(trip_count, exec_condition);
    loop->set_function(body);
    loop->set_special_body_ports(ngraph::opset5::Loop::SpecialBodyPorts{1, 1});

    loop->set_invariant_input(Xi, X);
    loop->set_invariant_input(Yi, Y);
    loop->set_merged_input(M_body, M, Zo);

    // check input descriptors
    for (auto& desc : loop->get_input_descriptions())
    {
        auto type_info = desc->get_type_info();
        if (std::strcmp(type_info.name, "InvariantInputDescription") == 0)
        {
            auto input_desc =
                as_type_ptr<ngraph::opset5::TensorIterator::InvariantInputDescription>(desc);
            EXPECT_NE(input_desc, nullptr);
        }
        else if (std::strcmp(type_info.name, "SliceInputDescription") == 0)
        {
            auto input_desc =
                as_type_ptr<ngraph::opset5::TensorIterator::SliceInputDescription>(desc);
            EXPECT_NE(input_desc, nullptr);
        }
        else if (std::strcmp(type_info.name, "MergedInputDescription") == 0)
        {
            auto input_desc =
                as_type_ptr<ngraph::opset5::TensorIterator::MergedInputDescription>(desc);
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
    for (auto& desc : loop->get_output_descriptions())
    {
        auto type_info = desc->get_type_info();
        if (std::strcmp(type_info.name, "ConcatOutputDescription") == 0)
        {
            auto output_desc =
                as_type_ptr<ngraph::opset5::TensorIterator::ConcatOutputDescription>(desc);
            EXPECT_NE(output_desc, nullptr);
        }
        else if (std::strcmp(type_info.name, "BodyOutputDescription") == 0)
        {
            auto output_desc =
                as_type_ptr<ngraph::opset5::TensorIterator::BodyOutputDescription>(desc);
            EXPECT_NE(output_desc, nullptr);
        }
    }
    auto result0 = make_shared<opset5::Result>(out0);
    auto result1 = make_shared<opset5::Result>(out1);
    auto result2 = make_shared<opset5::Result>(out2);
    Shape out0_shape{};
    Shape out1_shape{32, 1, 10};
    Shape out2_shape{32, 10, 10};

    auto results = ResultVector{result0, result1, result2};
    auto f = make_shared<Function>(results, ParameterVector{X, Y, M});
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
TEST(type_prop, loop_operation_10_iter_static_shapes_sliced_inputs)
{
    // That which we iterate over
    auto X = make_shared<opset5::Parameter>(element::f32, Shape{32, 1, 10});
    auto Y = make_shared<opset5::Parameter>(element::f32, Shape{32, 10, 1});
    auto M = make_shared<opset5::Parameter>(element::f32, Shape{32, 1, 10});

    // Set up the cell body, a function from (Xi, Yi) -> (Zo)
    // Body parameters
    auto current_iteration = make_shared<opset5::Parameter>(element::i64, Shape{});
    auto Xi = make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());
    auto Yi = make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());
    auto M_body = make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());
    auto body_condition =
        std::make_shared<ngraph::opset5::Constant>(ngraph::element::boolean, ngraph::Shape{}, true);

    auto trip_count =
        std::make_shared<ngraph::opset5::Constant>(ngraph::element::i64, ngraph::Shape{}, 10);
    auto exec_condition =
        std::make_shared<ngraph::opset5::Constant>(ngraph::element::boolean, ngraph::Shape{}, true);
    // Body
    auto sum = make_shared<ngraph::opset5::Add>(Xi, Yi);
    auto Zo = make_shared<ngraph::opset5::Multiply>(sum, M_body);
    auto body = make_shared<ngraph::Function>(OutputVector{Zo, body_condition, sum},
                                              ParameterVector{Xi, current_iteration, Yi, M_body});

    auto loop = make_shared<opset5::Loop>(trip_count, exec_condition);
    loop->set_function(body);
    loop->set_special_body_ports(ngraph::opset5::Loop::SpecialBodyPorts{1, 1});

    loop->set_sliced_input(Xi, X, 0, 1, 1, -1, 2);
    loop->set_sliced_input(Yi, Y, -1, -1, 1, 0, 1);
    loop->set_merged_input(M_body, M, Zo);

    // check input descriptors
    for (auto& desc : loop->get_input_descriptions())
    {
        auto type_info = desc->get_type_info();
        if (std::strcmp(type_info.name, "InvariantInputDescription") == 0)
        {
            auto input_desc =
                as_type_ptr<ngraph::opset5::TensorIterator::InvariantInputDescription>(desc);
            EXPECT_NE(input_desc, nullptr);
        }
        else if (std::strcmp(type_info.name, "SliceInputDescription") == 0)
        {
            auto input_desc =
                as_type_ptr<ngraph::opset5::TensorIterator::SliceInputDescription>(desc);
            EXPECT_NE(input_desc, nullptr);
        }
        else if (std::strcmp(type_info.name, "MergedInputDescription") == 0)
        {
            auto input_desc =
                as_type_ptr<ngraph::opset5::TensorIterator::MergedInputDescription>(desc);
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
    for (auto& desc : loop->get_output_descriptions())
    {
        auto type_info = desc->get_type_info();
        if (std::strcmp(type_info.name, "ConcatOutputDescription") == 0)
        {
            auto output_desc =
                as_type_ptr<ngraph::opset5::TensorIterator::ConcatOutputDescription>(desc);
            EXPECT_NE(output_desc, nullptr);
        }
        else if (std::strcmp(type_info.name, "BodyOutputDescription") == 0)
        {
            auto output_desc =
                as_type_ptr<ngraph::opset5::TensorIterator::BodyOutputDescription>(desc);
            EXPECT_NE(output_desc, nullptr);
        }
    }
    auto result0 = make_shared<opset5::Result>(out0);
    auto result1 = make_shared<opset5::Result>(out1);
    auto result2 = make_shared<opset5::Result>(out2);
    auto result3 = make_shared<opset5::Result>(out3);
    Shape out0_shape{};
    Shape out1_shape{32, 1, 10};
    Shape out2_shape{32, 10, 10};
    Shape out3_shape{32, 1, 1};

    auto results = ResultVector{result0, result1, result2, result3};
    auto f = make_shared<Function>(results, ParameterVector{X, Y, M});
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
TEST(type_prop, loop_operation_dynamic_iter_dynamic_batch_shapes_sliced_inputs_concatenated_outputs)
{
    // That which we iterate over
    auto X =
        make_shared<opset5::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 1, 10});
    auto Y =
        make_shared<opset5::Parameter>(element::f32, PartialShape{32, Dimension::dynamic(), 10});
    auto M = make_shared<opset5::Parameter>(element::f32, PartialShape{32, 1, 10});
    auto T = make_shared<opset5::Parameter>(element::i64, Shape{});

    // Set up the cell body, a function from (Xi, Yi) -> (Zo)
    // Body parameters
    auto current_iteration = make_shared<opset5::Parameter>(element::i64, Shape{});
    auto Xi = make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());
    auto Yi = make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());
    auto M_body = make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());

    auto body_condition = make_shared<opset5::Constant>(element::boolean, Shape{}, true);
    auto exec_condition = make_shared<opset5::Constant>(element::boolean, Shape{}, true);

    // Body
    auto sum = make_shared<opset5::Add>(Xi, Yi);
    auto Zo = make_shared<opset5::Multiply>(sum, M_body);
    auto body = make_shared<Function>(OutputVector{Zo, body_condition, sum},
                                      ParameterVector{Xi, Yi, current_iteration, M_body});

    auto loop = make_shared<opset5::Loop>(T, exec_condition);
    loop->set_function(body);
    loop->set_special_body_ports(opset5::Loop::SpecialBodyPorts{2, 1});

    loop->set_sliced_input(Xi, X, 0, 1, 1, -1, 0);
    loop->set_sliced_input(Yi, Y, -1, -1, 1, 0, 1);
    loop->set_merged_input(M_body, M, Zo);

    // check input descriptors
    for (auto& desc : loop->get_input_descriptions())
    {
        auto type_info = desc->get_type_info();
        if (std::strcmp(type_info.name, "InvariantInputDescription") == 0)
        {
            auto input_desc =
                as_type_ptr<ngraph::opset5::TensorIterator::InvariantInputDescription>(desc);
            EXPECT_NE(input_desc, nullptr);
        }
        else if (std::strcmp(type_info.name, "SliceInputDescription") == 0)
        {
            auto input_desc =
                as_type_ptr<ngraph::opset5::TensorIterator::SliceInputDescription>(desc);
            EXPECT_NE(input_desc, nullptr);
        }
        else if (std::strcmp(type_info.name, "MergedInputDescription") == 0)
        {
            auto input_desc =
                as_type_ptr<ngraph::opset5::TensorIterator::MergedInputDescription>(desc);
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
    for (auto& desc : loop->get_output_descriptions())
    {
        auto type_info = desc->get_type_info();
        if (std::strcmp(type_info.name, "ConcatOutputDescription") == 0)
        {
            auto output_desc =
                as_type_ptr<ngraph::opset5::TensorIterator::ConcatOutputDescription>(desc);
            EXPECT_NE(output_desc, nullptr);
        }
        else if (std::strcmp(type_info.name, "BodyOutputDescription") == 0)
        {
            auto output_desc =
                as_type_ptr<ngraph::opset5::TensorIterator::BodyOutputDescription>(desc);
            EXPECT_NE(output_desc, nullptr);
        }
    }
    auto result0 = make_shared<opset5::Result>(out0);
    auto result1 = make_shared<opset5::Result>(out1);
    auto result2 = make_shared<opset5::Result>(out2);
    auto result3 = make_shared<opset5::Result>(out3);
    Shape out0_shape{};
    Shape out1_shape{32, 1, 10};
    PartialShape out2_shape{32, Dimension::dynamic(), 10};
    Shape out3_shape{32, 1, 10};

    auto results = ResultVector{result0, result1, result2, result3};
    auto f = make_shared<Function>(results, ParameterVector{X, Y, T, M});
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
TEST(type_prop, loop_operation_dynamic_iter_dynamic_shapes_sliced_inputs_concatenated_outputs)
{
    // That which we iterate over
    auto X = make_shared<opset5::Parameter>(
        element::f32, PartialShape{Dimension::dynamic(), Dimension::dynamic(), 10});
    auto T = make_shared<opset5::Parameter>(element::i64, Shape{});

    // Set up the cell body, a function from (Xi, Yi) -> (Zo)
    // Body parameters
    auto current_iteration = make_shared<opset5::Parameter>(element::i64, Shape{});
    auto Xi = make_shared<opset5::Parameter>(element::f32, PartialShape::dynamic());

    auto body_condition = make_shared<opset5::Constant>(element::boolean, Shape{}, true);
    auto exec_condition = make_shared<opset5::Constant>(element::boolean, Shape{}, true);

    // Body
    auto Zo = make_shared<opset5::Multiply>(Xi, Xi);
    auto body = make_shared<Function>(OutputVector{Zo, body_condition},
                                      ParameterVector{Xi, current_iteration});

    auto loop = make_shared<opset5::Loop>(T, exec_condition);
    loop->set_function(body);
    loop->set_special_body_ports(opset5::Loop::SpecialBodyPorts{1, 1});

    loop->set_sliced_input(Xi, X, 0, 1, 1, -1, 0);

    // check input descriptors
    for (auto& desc : loop->get_input_descriptions())
    {
        auto type_info = desc->get_type_info();
        if (std::strcmp(type_info.name, "InvariantInputDescription") == 0)
        {
            auto input_desc =
                as_type_ptr<ngraph::opset5::TensorIterator::InvariantInputDescription>(desc);
            EXPECT_NE(input_desc, nullptr);
        }
        else if (std::strcmp(type_info.name, "SliceInputDescription") == 0)
        {
            auto input_desc =
                as_type_ptr<ngraph::opset5::TensorIterator::SliceInputDescription>(desc);
            EXPECT_NE(input_desc, nullptr);
        }
        else if (std::strcmp(type_info.name, "MergedInputDescription") == 0)
        {
            auto input_desc =
                as_type_ptr<ngraph::opset5::TensorIterator::MergedInputDescription>(desc);
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
    for (auto& desc : loop->get_output_descriptions())
    {
        auto type_info = desc->get_type_info();
        if (std::strcmp(type_info.name, "ConcatOutputDescription") == 0)
        {
            auto output_desc =
                as_type_ptr<ngraph::opset5::TensorIterator::ConcatOutputDescription>(desc);
            EXPECT_NE(output_desc, nullptr);
        }
        else if (std::strcmp(type_info.name, "BodyOutputDescription") == 0)
        {
            auto output_desc =
                as_type_ptr<ngraph::opset5::TensorIterator::BodyOutputDescription>(desc);
            EXPECT_NE(output_desc, nullptr);
        }
    }
    auto result0 = make_shared<opset5::Result>(out0);
    auto result1 = make_shared<opset5::Result>(out1);
    auto result2 = make_shared<opset5::Result>(out2);
    Shape out0_shape{};
    PartialShape out1_shape{1, Dimension::dynamic(), 10};
    PartialShape out2_shape{Dimension::dynamic(), Dimension::dynamic(), 10};

    auto results = ResultVector{result0, result1, result2};
    auto f = make_shared<Function>(results, ParameterVector{X, T});
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
