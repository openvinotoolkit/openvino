// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/if.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/opsets/opset10_decl.hpp"
#include "transformations/common_optimizations/push_constant_to_subgraph.hpp"

using namespace testing;
using namespace ov;

TEST_F(TransformationTestsF, PushConstantToSubgraphLoop) {
    {
        auto trip_count = opset10::Constant::create(element::i32, Shape{}, {2});
        auto term_cond = opset10::Constant::create(element::boolean, Shape{}, {true});
        std::shared_ptr<Model> loop_body;
        {
            auto X = std::make_shared<opset10::Parameter>(element::f32, Shape{1, 2});
            auto Y = std::make_shared<opset10::Parameter>(element::f32, Shape{1, 2});
            auto Z = std::make_shared<opset10::Parameter>(element::f32, Shape{1, 2});
            auto mul = std::make_shared<opset10::Multiply>(X, Y);
            auto add = std::make_shared<opset10::Add>(mul, Z);
            auto cond = opset10::Constant::create(element::boolean, Shape{}, {true});
            loop_body = std::make_shared<Model>(OutputVector{add, cond}, ParameterVector{X, Y, Z});
        }
        auto loop = std::make_shared<opset10::Loop>(trip_count, term_cond);
        loop->set_function(loop_body);

        auto X = std::make_shared<opset10::Parameter>(element::f32, Shape{2, 2});
        auto constant_1 = opset10::Constant::create(element::i32, Shape{2, 2}, {11});
        auto convert_1 = std::make_shared<opset10::Convert>(constant_1, element::f32);
        auto constant_2 = opset10::Constant::create(element::i32, Shape{1, 2}, {22});
        auto convert_2 = std::make_shared<opset10::Convert>(constant_2, element::f32);
        const auto& loop_params = loop_body->get_parameters();
        loop->set_special_body_ports({-1, 1});
        loop->set_sliced_input(loop_params[0], X, 0, 1, 1, -1, 0);
        loop->set_sliced_input(loop_params[1], convert_1, 0, 1, 1, -1, 0);
        loop->set_invariant_input(loop_params[2], convert_2);
        auto out = loop->get_concatenated_slices(loop_body->get_results()[0], 0, 1, 1, -1, 0);
        model = std::make_shared<Model>(OutputVector{out}, ParameterVector{X});

        manager.register_pass<pass::PushConstantToSubgraph>();
    }

    {
        auto trip_count = opset10::Constant::create(element::i32, Shape{}, {2});
        auto term_cond = opset10::Constant::create(element::boolean, Shape{}, {true});
        std::shared_ptr<Model> loop_body;
        {
            auto constant = opset10::Constant::create(element::f32, Shape{1, 2}, {22});
            auto X = std::make_shared<opset10::Parameter>(element::f32, Shape{1, 2});
            auto Y = std::make_shared<opset10::Parameter>(element::f32, Shape{1, 2});
            auto mul = std::make_shared<opset10::Multiply>(X, Y);
            auto add = std::make_shared<opset10::Add>(mul, constant);
            auto cond = opset10::Constant::create(element::boolean, Shape{}, {true});
            loop_body = std::make_shared<Model>(OutputVector{add, cond}, ParameterVector{X, Y});
        }
        auto loop = std::make_shared<opset10::Loop>(trip_count, term_cond);
        loop->set_function(loop_body);

        auto X = std::make_shared<opset10::Parameter>(element::f32, Shape{2, 2});
        auto constant_1 = opset10::Constant::create(element::i32, Shape{2, 2}, {11});
        auto convert_1 = std::make_shared<opset10::Convert>(constant_1, element::f32);
        const auto& loop_params = loop_body->get_parameters();
        loop->set_special_body_ports({-1, 1});
        loop->set_sliced_input(loop_params[0], X, 0, 1, 1, -1, 0);
        loop->set_sliced_input(loop_params[1], convert_1, 0, 1, 1, -1, 0);
        auto out = loop->get_concatenated_slices(loop_body->get_results()[0], 0, 1, 1, -1, 0);
        model_ref = std::make_shared<Model>(OutputVector{out}, ParameterVector{X});
    }
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, PushConstantToSubgraphIf) {
    {
        auto cond = opset10::Constant::create(element::boolean, Shape{}, {false});
        auto if_op = std::make_shared<ov::opset10::If>(cond);
        std::shared_ptr<ov::Model> then_body;
        {
            auto A = std::make_shared<ov::opset10::Parameter>(ov::element::f32, ov::Shape{3});
            auto B = std::make_shared<ov::opset10::Parameter>(ov::element::f32, ov::Shape{3});
            auto C = std::make_shared<ov::opset10::Parameter>(ov::element::f32, ov::Shape{3});
            auto D = std::make_shared<ov::opset10::Parameter>(ov::element::f32, ov::Shape{3});
            auto add = std::make_shared<ov::opset10::Add>(A, B);
            auto mul = std::make_shared<ov::opset10::Multiply>(add, C);
            auto sub = std::make_shared<ov::opset10::Subtract>(mul, D);
            then_body = std::make_shared<ov::Model>(add, ov::ParameterVector{A, B, C, D});
        }
        std::shared_ptr<ov::Model> else_body;
        {
            auto A = std::make_shared<ov::opset10::Parameter>(ov::element::f32, ov::Shape{3});
            auto B = std::make_shared<ov::opset10::Parameter>(ov::element::f32, ov::Shape{3});
            auto C = std::make_shared<ov::opset10::Parameter>(ov::element::f32, ov::Shape{3});
            auto D = std::make_shared<ov::opset10::Parameter>(ov::element::f32, ov::Shape{3});
            auto mul = std::make_shared<ov::opset10::Multiply>(A, B);
            auto add = std::make_shared<ov::opset10::Add>(mul, C);
            auto div = std::make_shared<ov::opset10::Divide>(add, D);
            else_body = std::make_shared<ov::Model>(div, ov::ParameterVector{A, B, C, D});
        }

        if_op->set_then_body(then_body);
        if_op->set_else_body(else_body);

        const auto& then_params = then_body->get_parameters();
        const auto& else_params = else_body->get_parameters();

        auto A = std::make_shared<ov::opset10::Parameter>(ov::element::f32, ov::Shape{3});
        auto B = std::make_shared<ov::opset10::Parameter>(ov::element::f32, ov::Shape{3});
        auto C = std::make_shared<ov::opset10::Parameter>(ov::element::f32, ov::Shape{3});
        auto const_1 = ov::opset10::Constant::create(ov::element::i32, ov::Shape{3}, {1});
        auto convert_1 = std::make_shared<ov::opset10::Convert>(const_1, ov::element::f32);
        auto const_2 = ov::opset10::Constant::create(ov::element::i32, ov::Shape{3}, {2});
        auto convert_2 = std::make_shared<ov::opset10::Convert>(const_2, ov::element::f32);
        auto const_3 = ov::opset10::Constant::create(ov::element::i32, ov::Shape{3}, {3});
        auto convert_3 = std::make_shared<ov::opset10::Convert>(const_3, ov::element::f32);

        if_op->set_input(A, then_params[0], nullptr);
        if_op->set_input(convert_1, then_params[1], nullptr);
        if_op->set_input(B, then_params[2], else_params[0]);
        if_op->set_input(convert_2, then_params[3], else_params[1]);

        if_op->set_input(C, nullptr, else_params[2]);
        if_op->set_input(convert_3, nullptr, else_params[3]);
        if_op->set_output(then_body->get_results()[0], else_body->get_results()[0]);

        model = std::make_shared<ov::Model>(if_op, ov::ParameterVector{A, B, C});

        manager.register_pass<pass::PushConstantToSubgraph>();
    }

    {
        auto cond = opset10::Constant::create(element::boolean, Shape{}, {false});
        auto const_1 = ov::opset10::Constant::create(ov::element::f32, ov::Shape{3}, {1});
        auto const_2 = ov::opset10::Constant::create(ov::element::f32, ov::Shape{3}, {2});
        auto const_3 = ov::opset10::Constant::create(ov::element::f32, ov::Shape{3}, {3});
        auto if_op = std::make_shared<ov::opset10::If>(cond);
        std::shared_ptr<ov::Model> then_body;
        {
            auto A = std::make_shared<ov::opset10::Parameter>(ov::element::f32, ov::Shape{3});
            auto B = std::make_shared<ov::opset10::Parameter>(ov::element::f32, ov::Shape{3});
            auto add = std::make_shared<ov::opset10::Add>(A, const_1);
            auto mul = std::make_shared<ov::opset10::Multiply>(add, B);
            auto sub = std::make_shared<ov::opset10::Subtract>(mul, const_2);
            then_body = std::make_shared<ov::Model>(add, ov::ParameterVector{A, B});
        }
        std::shared_ptr<ov::Model> else_body;
        {
            auto A = std::make_shared<ov::opset10::Parameter>(ov::element::f32, ov::Shape{3});
            auto B = std::make_shared<ov::opset10::Parameter>(ov::element::f32, ov::Shape{3});
            auto mul = std::make_shared<ov::opset10::Multiply>(A, const_2);
            auto add = std::make_shared<ov::opset10::Add>(mul, B);
            auto div = std::make_shared<ov::opset10::Divide>(add, const_3);
            else_body = std::make_shared<ov::Model>(div, ov::ParameterVector{A, B});
        }

        if_op->set_then_body(then_body);
        if_op->set_else_body(else_body);

        const auto& then_params = then_body->get_parameters();
        const auto& else_params = else_body->get_parameters();

        auto A = std::make_shared<ov::opset10::Parameter>(ov::element::f32, ov::Shape{3});
        auto B = std::make_shared<ov::opset10::Parameter>(ov::element::f32, ov::Shape{3});
        auto C = std::make_shared<ov::opset10::Parameter>(ov::element::f32, ov::Shape{3});

        if_op->set_input(A, then_params[0], nullptr);
        if_op->set_input(B, then_params[1], else_params[0]);
        if_op->set_input(C, nullptr, else_params[1]);
        if_op->set_output(then_body->get_results()[0], else_body->get_results()[0]);

        model_ref = std::make_shared<ov::Model>(if_op, ov::ParameterVector{A, B, C});
    }

    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, PushConstantToSubgraphLoopMoreThan32Inputs) {
    int num_const_inputs = 33;
    {
        auto trip_count = opset10::Constant::create(element::i32, Shape{}, {2});
        auto term_cond = opset10::Constant::create(element::boolean, Shape{}, {true});
        std::shared_ptr<Model> loop_body;
        {
            auto X = std::make_shared<opset10::Parameter>(element::f32, Shape{1, 2});
            ParameterVector params;
            params.reserve(num_const_inputs + 1);
            params.push_back(X);
            NodeVector concat_inputs;
            concat_inputs.reserve(num_const_inputs + 1);
            concat_inputs.push_back(X);
            for (int i = 0; i < num_const_inputs; i++) {
                params.push_back(std::make_shared<opset10::Parameter>(element::f32, Shape{1, 2}));
                concat_inputs.push_back(params.back());
            }
            auto concat = std::make_shared<opset10::Concat>(concat_inputs, 1);
            auto cond = opset10::Constant::create(element::boolean, Shape{}, {true});
            loop_body = std::make_shared<Model>(OutputVector{concat, cond}, params);
        }
        auto loop = std::make_shared<opset10::Loop>(trip_count, term_cond);
        loop->set_function(loop_body);

        auto X = std::make_shared<opset10::Parameter>(element::f32, Shape{2, 2});
        NodeVector constants;
        constants.reserve(num_const_inputs);
        for (int i = 0; i < num_const_inputs; i++) {
            constants.push_back(opset10::Constant::create(element::f32, Shape{1, 2}, {-2}));
        }
        const auto& loop_params = loop_body->get_parameters();
        loop->set_special_body_ports({-1, 1});
        loop->set_sliced_input(loop_params[0], X, 0, 1, 1, -1, 0);
        for (int i = 0; i < num_const_inputs; i++) {
            loop->set_invariant_input(loop_params[i + 1], constants[i]);
        }
        auto out = loop->get_concatenated_slices(loop_body->get_results()[0], 0, 1, 1, -1, 0);
        model = std::make_shared<Model>(OutputVector{out}, ParameterVector{X});

        manager.register_pass<pass::PushConstantToSubgraph>();
    }

    {
        auto trip_count = opset10::Constant::create(element::i32, Shape{}, {2});
        auto term_cond = opset10::Constant::create(element::boolean, Shape{}, {true});
        std::shared_ptr<Model> loop_body;
        {
            auto constant = opset10::Constant::create(element::f32, Shape{1, 2}, {-2});
            auto X = std::make_shared<opset10::Parameter>(element::f32, Shape{1, 2});
            NodeVector concat_inputs;
            concat_inputs.reserve(num_const_inputs + 1);
            concat_inputs.push_back(X);
            for (int i = 0; i < num_const_inputs; i++) {
                concat_inputs.push_back(opset10::Constant::create(element::f32, Shape{1, 2}, {-2}));
            }
            auto concat = std::make_shared<opset10::Concat>(concat_inputs, 1);
            auto cond = opset10::Constant::create(element::boolean, Shape{}, {true});
            loop_body = std::make_shared<Model>(OutputVector{concat, cond}, ParameterVector{X});
        }
        auto loop = std::make_shared<opset10::Loop>(trip_count, term_cond);
        loop->set_function(loop_body);

        auto X = std::make_shared<opset10::Parameter>(element::f32, Shape{2, 2});
        const auto& loop_params = loop_body->get_parameters();
        loop->set_special_body_ports({-1, 1});
        loop->set_sliced_input(loop_params[0], X, 0, 1, 1, -1, 0);
        auto out = loop->get_concatenated_slices(loop_body->get_results()[0], 0, 1, 1, -1, 0);
        model_ref = std::make_shared<Model>(OutputVector{out}, ParameterVector{X});
    }
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}
