// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/fold_subgraph_empty_inputs.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"

using namespace testing;
using namespace ov;
using namespace ov::opset8;

TEST_F(TransformationTestsF, FoldLoopEmptyInputs) {
    auto trip_count = std::make_shared<Constant>(element::i64, Shape{}, 10);
    auto condition = std::make_shared<Constant>(element::boolean, Shape{}, true);

    auto a = std::make_shared<Parameter>(element::f32, Shape{2, 0});
    auto a_add = std::make_shared<Add>(a, a);
    auto ai = std::make_shared<Parameter>(element::f32, Shape{2, 0});

    auto mul = std::make_shared<Multiply>(ai, ai);
    auto abs = std::make_shared<Abs>(mul);
    {
        auto body = std::make_shared<Model>(OutputVector{condition, abs}, ParameterVector{ai});
        auto loop = std::make_shared<Loop>(trip_count, condition);
        loop->set_special_body_ports({-1, 0});
        loop->set_function(body);
        loop->set_invariant_input(ai, a_add);

        auto loop_res = std::make_shared<Result>(loop->get_iter_value(abs));
        model = std::make_shared<Model>(OutputVector{loop_res}, ParameterVector{a});

        manager.register_pass<ov::pass::FoldSubgraphEmptyInputs>();
    }
    {
        auto body = std::make_shared<Model>(OutputVector{condition, abs}, ParameterVector{ai});
        auto loop = std::make_shared<Loop>(trip_count, condition);
        loop->set_special_body_ports({-1, 0});
        loop->set_function(body);
        const auto const_input = std::make_shared<Constant>(a_add->get_element_type(), a_add->get_shape());
        loop->set_invariant_input(ai, const_input);

        auto loop_res = std::make_shared<Result>(loop->get_iter_value(abs));
        model_ref = std::make_shared<Model>(OutputVector{loop_res}, ParameterVector{a});
    }
}

TEST_F(TransformationTestsF, FoldLoopManyEmptyInputs) {
    auto trip_count = std::make_shared<Constant>(element::i64, Shape{}, 10);
    auto condition = std::make_shared<Constant>(element::boolean, Shape{}, true);

    auto a = std::make_shared<Parameter>(element::f32, Shape{2, 0});
    auto a_add = std::make_shared<Add>(a, a);
    auto ai = std::make_shared<Parameter>(element::f32, Shape{2, 0});

    auto b = std::make_shared<Parameter>(element::f32, Shape{2, 2});
    auto bi = std::make_shared<Parameter>(element::f32, Shape{2, 2});
    auto b_add = std::make_shared<Add>(b, b);

    auto c = std::make_shared<Parameter>(element::f32, Shape{2, 0});
    auto c_add = std::make_shared<Add>(c, c);
    auto ci = std::make_shared<Parameter>(element::f32, Shape{2, 0});

    auto concat = std::make_shared<Concat>(OutputVector{ai, bi, ci}, 1);
    {
        auto body = std::make_shared<Model>(OutputVector{condition, concat}, ParameterVector{ai, bi, ci});
        auto loop = std::make_shared<Loop>(trip_count, condition);
        loop->set_special_body_ports({-1, 0});
        loop->set_function(body);
        loop->set_invariant_input(ai, a_add);
        loop->set_invariant_input(bi, b_add);
        loop->set_invariant_input(ci, c_add);

        auto loop_res = std::make_shared<Result>(loop->get_iter_value(concat));
        model = std::make_shared<Model>(OutputVector{loop_res}, ParameterVector{a, b, c});

        manager.register_pass<ov::pass::FoldSubgraphEmptyInputs>();
    }
    {
        auto body = std::make_shared<Model>(OutputVector{condition, concat}, ParameterVector{ai, bi, ci});
        auto loop = std::make_shared<Loop>(trip_count, condition);
        loop->set_special_body_ports({-1, 0});
        loop->set_function(body);
        loop->set_invariant_input(ai, std::make_shared<Constant>(a_add->get_element_type(), a_add->get_shape()));
        loop->set_invariant_input(bi, b_add);
        loop->set_invariant_input(ci, std::make_shared<Constant>(c_add->get_element_type(), c_add->get_shape()));

        auto loop_res = std::make_shared<Result>(loop->get_iter_value(concat));
        model_ref = std::make_shared<Model>(OutputVector{loop_res}, ParameterVector{b});
    }
}

TEST_F(TransformationTestsF, FoldLoopEmptyMergedInputs) {
    auto trip_count = std::make_shared<Constant>(element::i64, Shape{}, 10);
    auto condition = std::make_shared<Constant>(element::boolean, Shape{}, true);

    auto x_init = std::make_shared<Parameter>(element::f32, Shape{0, 10});
    auto xi = std::make_shared<Parameter>(element::f32, Shape{0, 10});
    auto x_add = std::make_shared<Add>(x_init, x_init);

    auto y_const = std::make_shared<Constant>(element::f32, Shape{1, 10});

    auto concat = std::make_shared<Concat>(OutputVector{xi, y_const}, 0);
    {
        auto body = std::make_shared<Model>(OutputVector{condition, concat}, ParameterVector{xi});
        auto loop = std::make_shared<Loop>(trip_count, condition);
        loop->set_special_body_ports({-1, 0});
        loop->set_function(body);
        loop->set_merged_input(xi, x_add, concat);
        auto loop_res = std::make_shared<Result>(loop->get_iter_value(concat));
        model = std::make_shared<Model>(OutputVector{loop_res}, ParameterVector{x_init});

        manager.register_pass<ov::pass::FoldSubgraphEmptyInputs>();
    }
    {
        auto body = std::make_shared<Model>(OutputVector{condition, concat}, ParameterVector{xi});
        auto loop = std::make_shared<Loop>(trip_count, condition);
        loop->set_special_body_ports({-1, 0});
        loop->set_function(body);
        auto x_add_const = std::make_shared<Constant>(x_add->get_element_type(), x_add->get_shape());
        loop->set_merged_input(xi, x_add_const, concat);
        auto loop_res = std::make_shared<Result>(loop->get_iter_value(concat));
        model_ref = std::make_shared<Model>(OutputVector{loop_res}, ParameterVector{});
    }
}

TEST_F(TransformationTestsF, FoldLoopSkipEmptyConstants) {
    auto trip_count = std::make_shared<Constant>(element::i64, Shape{}, 10);
    auto condition = std::make_shared<Constant>(element::boolean, Shape{}, true);

    auto a = std::make_shared<Constant>(element::f32, Shape{2, 0});
    auto ai = std::make_shared<Parameter>(element::f32, Shape{2, 0});

    auto mul = std::make_shared<Multiply>(ai, ai);
    auto abs = std::make_shared<Abs>(mul);

    auto body = std::make_shared<Model>(OutputVector{condition, abs}, ParameterVector{ai});
    auto loop = std::make_shared<Loop>(trip_count, condition);
    loop->set_special_body_ports({-1, 0});
    loop->set_function(body);
    loop->set_invariant_input(ai, a);

    auto loop_res = std::make_shared<Result>(loop->get_iter_value(abs));
    model = std::make_shared<Model>(OutputVector{loop_res}, ParameterVector{});

    manager.register_pass<ov::pass::FoldSubgraphEmptyInputs>();
}

TEST_F(TransformationTestsF, FoldLoopSkipDynamicInputs) {
    auto trip_count = std::make_shared<Constant>(element::i64, Shape{}, 10);
    auto condition = std::make_shared<Constant>(element::boolean, Shape{}, true);

    auto a = std::make_shared<Parameter>(element::f32, PartialShape{2, Dimension::dynamic()});
    auto a_add = std::make_shared<Add>(a, a);
    auto ai = std::make_shared<Parameter>(element::f32, Shape{2, 0});

    auto mul = std::make_shared<Multiply>(ai, ai);
    auto abs = std::make_shared<Abs>(mul);

    auto body = std::make_shared<Model>(OutputVector{condition, abs}, ParameterVector{ai});
    auto loop = std::make_shared<Loop>(trip_count, condition);
    loop->set_special_body_ports({-1, 0});
    loop->set_function(body);
    loop->set_invariant_input(ai, a_add);

    auto loop_res = std::make_shared<Result>(loop->get_iter_value(abs));
    model = std::make_shared<Model>(OutputVector{loop_res}, ParameterVector{a});

    manager.register_pass<ov::pass::FoldSubgraphEmptyInputs>();
}

TEST_F(TransformationTestsF, FoldLoopSkipNonEmptyInputs) {
    auto trip_count = std::make_shared<Constant>(element::i64, Shape{}, 10);
    auto condition = std::make_shared<Constant>(element::boolean, Shape{}, true);

    auto a = std::make_shared<Parameter>(element::f32, Shape{2, 2});
    auto a_add = std::make_shared<Add>(a, a);
    auto ai = std::make_shared<Parameter>(element::f32, Shape{2, 2});

    auto mul = std::make_shared<Multiply>(ai, ai);
    auto abs = std::make_shared<Abs>(mul);

    auto body = std::make_shared<Model>(OutputVector{condition, abs}, ParameterVector{ai});
    auto loop = std::make_shared<Loop>(trip_count, condition);
    loop->set_special_body_ports({-1, 0});
    loop->set_function(body);
    loop->set_invariant_input(ai, a_add);

    auto loop_res = std::make_shared<Result>(loop->get_iter_value(abs));
    model = std::make_shared<Model>(OutputVector{loop_res}, ParameterVector{a});

    manager.register_pass<ov::pass::FoldSubgraphEmptyInputs>();
}

TEST_F(TransformationTestsF, FoldIfManyEmptyInputs) {
    auto X = std::make_shared<Parameter>(element::f32, Shape{2, 0, 1});
    auto X_add = std::make_shared<Add>(X, X);
    auto Z = std::make_shared<Parameter>(element::f32, Shape{2, 0, 1});
    auto cond = std::make_shared<Constant>(element::boolean, Shape{1}, true);

    auto Xt = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    auto Zt = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());

    auto then_op = std::make_shared<Add>(Zt, Zt);
    auto then_op_res = std::make_shared<Result>(then_op);

    auto Xe = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    auto Ze = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());

    auto else_op = std::make_shared<Add>(std::make_shared<Maximum>(Xe, Ze), Ze);
    auto else_op_res = std::make_shared<Result>(else_op);
    {
        auto then_body = std::make_shared<Model>(OutputVector{then_op_res}, ParameterVector{Zt});
        auto else_body = std::make_shared<Model>(OutputVector{else_op_res}, ParameterVector{Xe, Ze});
        auto if_op = std::make_shared<If>(cond);
        if_op->set_then_body(then_body);
        if_op->set_else_body(else_body);
        if_op->set_input(X_add, nullptr, Xe);
        if_op->set_input(Z, Zt, Ze);
        auto res = if_op->set_output(then_op_res, else_op_res);
        model = std::make_shared<Model>(OutputVector{res}, ParameterVector{X, Z});

        manager.register_pass<ov::pass::FoldSubgraphEmptyInputs>();
    }
    {
        auto then_body = std::make_shared<Model>(OutputVector{then_op_res}, ParameterVector{Zt});
        auto else_body = std::make_shared<Model>(OutputVector{else_op_res}, ParameterVector{Xe, Ze});
        auto if_op = std::make_shared<If>(cond);
        if_op->set_then_body(then_body);
        if_op->set_else_body(else_body);
        const auto X_add_folded = std::make_shared<Constant>(X_add->get_element_type(), X_add->get_shape());
        if_op->set_input(X_add_folded, nullptr, Xe);
        const auto Z_folded = std::make_shared<Constant>(Z->get_element_type(), Z->get_shape());
        if_op->set_input(Z_folded, Zt, Ze);
        auto res = if_op->set_output(then_op_res, else_op_res);
        model_ref = std::make_shared<Model>(OutputVector{res}, ParameterVector{});
    }
}
