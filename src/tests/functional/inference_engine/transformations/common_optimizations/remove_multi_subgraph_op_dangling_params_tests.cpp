// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <openvino/core/model.hpp>
#include <openvino/opsets/opset8.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/common_optimizations/remove_multi_subgraph_op_dangling_params.hpp>
#include <transformations/common_optimizations/remove_concat_zero_dim_input.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ov;
using namespace ov::opset8;

TEST_F(TransformationTestsF, RemoveLoopDanglingParameters) {
    auto trip_count = std::make_shared<Constant>(element::i64, Shape{}, 10);
    auto condition = std::make_shared<Constant>(element::boolean, Shape{}, true);

    auto a = std::make_shared<Parameter>(element::f32, Shape{2, 2});
    auto ai = std::make_shared<Parameter>(element::f32, Shape{2, 2});
    auto b = std::make_shared<Parameter>(element::f32, Shape{2, 2});
    auto bi = std::make_shared<Parameter>(element::f32, Shape{2, 2});

    auto mul = std::make_shared<Multiply>(bi, bi);
    auto abs = std::make_shared<Abs>(mul);
    {
        auto body = std::make_shared<Model>(OutputVector{condition, abs}, ParameterVector{ai, bi});
        auto loop = std::make_shared<Loop>(trip_count, condition);
        loop->set_special_body_ports({-1, 0});
        loop->set_function(body);
        loop->set_invariant_input(ai, a);
        loop->set_invariant_input(bi, b);

        auto loop_res = std::make_shared<Result>(loop->get_iter_value(abs));
        function = std::make_shared<Model>(OutputVector{loop_res}, ParameterVector{a, b});

        manager.register_pass<pass::RemoveMultiSubGraphOpDanglingParams>();
    }
    {
        auto body = std::make_shared<Model>(OutputVector{condition, abs}, ParameterVector{bi});
        auto loop = std::make_shared<Loop>(trip_count, condition);
        loop->set_special_body_ports({-1, 0});
        loop->set_function(body);
        loop->set_invariant_input(bi, b);

        auto loop_res = std::make_shared<Result>(loop->get_iter_value(abs));
        function_ref = std::make_shared<Model>(OutputVector{loop_res}, ParameterVector{a, b});
    }
}

TEST_F(TransformationTestsF, RemoveLoopManyDanglingParameters) {
    auto trip_count = std::make_shared<Constant>(element::i64, Shape{}, 10);
    auto condition = std::make_shared<Constant>(element::boolean, Shape{}, true);

    auto a = std::make_shared<Parameter>(element::f32, Shape{2, 2});
    auto ai = std::make_shared<Parameter>(element::f32, Shape{2, 2});
    auto b = std::make_shared<Parameter>(element::f32, Shape{2, 2});
    auto bi = std::make_shared<Parameter>(element::f32, Shape{2, 2});
    auto c = std::make_shared<Parameter>(element::f32, Shape{2, 2});
    auto ci = std::make_shared<Parameter>(element::f32, Shape{2, 2});

    auto mul = std::make_shared<Multiply>(bi, bi);
    auto abs = std::make_shared<Abs>(mul);
    {
        auto body = std::make_shared<Model>(OutputVector{condition, abs}, ParameterVector{ai, bi, ci});
        auto loop = std::make_shared<Loop>(trip_count, condition);
        loop->set_special_body_ports({-1, 0});
        loop->set_function(body);
        loop->set_invariant_input(ai, a);
        loop->set_invariant_input(bi, b);
        loop->set_invariant_input(ci, c);

        auto loop_res = std::make_shared<Result>(loop->get_iter_value(abs));
        function = std::make_shared<Model>(OutputVector{loop_res}, ParameterVector{a, b, c});

        manager.register_pass<pass::RemoveMultiSubGraphOpDanglingParams>();
    }
    {
        auto body = std::make_shared<Model>(OutputVector{condition, abs}, ParameterVector{bi});
        auto loop = std::make_shared<Loop>(trip_count, condition);
        loop->set_special_body_ports({-1, 0});
        loop->set_function(body);
        loop->set_invariant_input(bi, b);

        auto loop_res = std::make_shared<Result>(loop->get_iter_value(abs));
        function_ref = std::make_shared<Model>(OutputVector{loop_res}, ParameterVector{a, b, c});
    }
}

TEST_F(TransformationTestsF, RemoveLoopManyDanglingParameters2) {
    auto trip_count = std::make_shared<Constant>(element::i64, Shape{}, 10);
    auto condition = std::make_shared<Constant>(element::boolean, Shape{}, true);

    auto a = std::make_shared<Parameter>(element::f32, Shape{2, 2});
    auto ai = std::make_shared<Parameter>(element::f32, Shape{2, 2});
    auto b = std::make_shared<Parameter>(element::f32, Shape{2, 2});
    auto bi = std::make_shared<Parameter>(element::f32, Shape{2, 2});
    auto c = std::make_shared<Parameter>(element::f32, Shape{2, 2});
    auto ci = std::make_shared<Parameter>(element::f32, Shape{2, 2});
    auto d = std::make_shared<Parameter>(element::f32, Shape{2, 2});
    auto di = std::make_shared<Parameter>(element::f32, Shape{2, 2});

    auto mul = std::make_shared<Multiply>(bi, bi);
    auto sub = std::make_shared<Multiply>(mul, di);
    auto abs = std::make_shared<Abs>(sub);
    {
        auto body = std::make_shared<Model>(OutputVector{condition, abs}, ParameterVector{ai, bi, ci, di});
        auto loop = std::make_shared<Loop>(trip_count, condition);
        loop->set_special_body_ports({-1, 0});
        loop->set_function(body);
        loop->set_invariant_input(ai, a);
        loop->set_invariant_input(bi, b);
        loop->set_invariant_input(ci, c);
        loop->set_invariant_input(di, d);

        auto loop_res = std::make_shared<Result>(loop->get_iter_value(abs));
        function = std::make_shared<Model>(OutputVector{loop_res}, ParameterVector{a, b, c, d});

        manager.register_pass<pass::RemoveMultiSubGraphOpDanglingParams>();
    }
    {
        auto body = std::make_shared<Model>(OutputVector{condition, abs}, ParameterVector{bi, di});
        auto loop = std::make_shared<Loop>(trip_count, condition);
        loop->set_special_body_ports({-1, 0});
        loop->set_function(body);
        loop->set_invariant_input(bi, b);
        loop->set_invariant_input(di, d);

        auto loop_res = std::make_shared<Result>(loop->get_iter_value(abs));
        function_ref = std::make_shared<Model>(OutputVector{loop_res}, ParameterVector{a, b, c, d});
    }
}

TEST_F(TransformationTestsF, RemoveLoopDanglingParametersIfConcatEmptyTensor) {
    auto trip_count = std::make_shared<Constant>(element::i64, Shape{}, 10);
    auto condition = std::make_shared<Constant>(element::boolean, Shape{}, true);

    auto a = std::make_shared<Parameter>(element::f32, Shape{2, 2});
    auto ai = std::make_shared<Parameter>(element::f32, Shape{2, 2});
    auto b = std::make_shared<Parameter>(element::f32, Shape{0, 2}); // empty tensor
    auto bi = std::make_shared<Parameter>(element::f32, Shape{0, 2});
    {
        auto concat = std::make_shared<Concat>(NodeVector{ai, bi}, 0);
        auto body = std::make_shared<Model>(OutputVector{condition, concat}, ParameterVector{ai, bi});
        auto loop = std::make_shared<Loop>(trip_count, condition);
        loop->set_special_body_ports({-1, 0});
        loop->set_function(body);
        loop->set_invariant_input(ai, a);
        loop->set_invariant_input(bi, b);

        auto loop_res = std::make_shared<Result>(loop->get_iter_value(concat));
        function = std::make_shared<Model>(OutputVector{loop_res}, ParameterVector{a, b});

        manager.register_pass<pass::RemoveConcatZeroDimInput>();
        manager.register_pass<pass::RemoveMultiSubGraphOpDanglingParams>();
    }
    {
        auto concat = std::make_shared<Concat>(NodeVector{ai}, 0);
        auto body = std::make_shared<Model>(OutputVector{condition, concat}, ParameterVector{ai});
        auto loop = std::make_shared<Loop>(trip_count, condition);
        loop->set_special_body_ports({-1, 0});
        loop->set_function(body);
        loop->set_invariant_input(ai, a);

        auto loop_res = std::make_shared<Result>(loop->get_iter_value(concat));
        function_ref = std::make_shared<Model>(OutputVector{loop_res}, ParameterVector{a, b});
    }
}

TEST_F(TransformationTestsF, RemoveIfDanglingParametersFromBodiesAndInputs) {
    auto X = std::make_shared<Parameter>(element::f32, Shape{2, 4, 1});
    auto Y = std::make_shared<Parameter>(element::f32, Shape{3, 4, 1});
    auto cond = std::make_shared<Constant>(element::boolean, Shape{1}, true);

    auto Xte = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    auto Yte = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());

    auto then_op = std::make_shared<Add>(Xte, Xte);
    auto then_op_res = std::make_shared<Result>(then_op);

    auto else_op = std::make_shared<Maximum>(Xte, Xte);
    auto else_op_res = std::make_shared<Result>(else_op);
    {
        auto then_body = std::make_shared<Model>(OutputVector{then_op_res}, ParameterVector{Xte, Yte});
        auto else_body = std::make_shared<Model>(OutputVector{else_op_res}, ParameterVector{Xte, Yte});
        auto if_op = std::make_shared<If>(cond);
        if_op->set_then_body(then_body);
        if_op->set_else_body(else_body);
        if_op->set_input(X, Xte, Xte);
        if_op->set_input(Y, Yte, Yte);
        auto res = if_op->set_output(then_op_res, else_op_res);
        function = std::make_shared<Model>(OutputVector{res}, ParameterVector{X, Y});

        manager.register_pass<pass::RemoveMultiSubGraphOpDanglingParams>();
    }
    {
        auto then_body = std::make_shared<Model>(OutputVector{then_op_res}, ParameterVector{Xte});
        auto else_body = std::make_shared<Model>(OutputVector{else_op_res}, ParameterVector{Xte});
        auto if_op = std::make_shared<If>(cond);
        if_op->set_then_body(then_body);
        if_op->set_else_body(else_body);
        if_op->set_input(X, Xte, Xte);
        auto res = if_op->set_output(then_op_res, else_op_res);
        function_ref = std::make_shared<Model>(OutputVector{res}, ParameterVector{X, Y});
    }
}

TEST_F(TransformationTestsF, RemoveIfDanglingParametersOnlyFromBodies) {
    auto X = std::make_shared<Parameter>(element::f32, Shape{2, 4, 1});
    auto Y = std::make_shared<Parameter>(element::f32, Shape{3, 4, 1});
    auto cond = std::make_shared<Constant>(element::boolean, Shape{1}, true);

    auto Xt = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    auto Yt = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());

    auto then_op = std::make_shared<Add>(Xt, Xt);
    auto then_op_res = std::make_shared<Result>(then_op);

    auto Xe = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    auto Ye = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());

    auto else_op = std::make_shared<Maximum>(Ye, Ye);
    auto else_op_res = std::make_shared<Result>(else_op);
    {
        auto then_body = std::make_shared<Model>(OutputVector{then_op_res}, ParameterVector{Xt, Yt});
        auto else_body = std::make_shared<Model>(OutputVector{else_op_res}, ParameterVector{Xe, Ye});
        auto if_op = std::make_shared<If>(cond);
        if_op->set_then_body(then_body);
        if_op->set_else_body(else_body);
        if_op->set_input(X, Xt, Xe);
        if_op->set_input(Y, Yt, Ye);
        auto res = if_op->set_output(then_op_res, else_op_res);
        function = std::make_shared<Model>(OutputVector{res}, ParameterVector{X, Y});

        manager.register_pass<pass::RemoveMultiSubGraphOpDanglingParams>();
    }
    {
        auto then_body = std::make_shared<Model>(OutputVector{then_op_res}, ParameterVector{Xt});
        auto else_body = std::make_shared<Model>(OutputVector{else_op_res}, ParameterVector{Ye});
        auto if_op = std::make_shared<If>(cond);
        if_op->set_then_body(then_body);
        if_op->set_else_body(else_body);
        if_op->set_input(X, Xt, nullptr);
        if_op->set_input(Y, nullptr, Ye);
        auto res = if_op->set_output(then_op_res, else_op_res);
        function_ref = std::make_shared<Model>(OutputVector{res}, ParameterVector{X, Y});
    }
}

TEST_F(TransformationTestsF, RemoveIfManyDanglingParameters) {
    auto X = std::make_shared<Parameter>(element::f32, Shape{2, 4, 1});
    auto Y = std::make_shared<Parameter>(element::f32, Shape{3, 4, 1});
    auto Z = std::make_shared<Parameter>(element::f32, Shape{2, 4, 1});
    auto cond = std::make_shared<Constant>(element::boolean, Shape{1}, true);

    auto Xt = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    auto Yt = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    auto Zt = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());

    auto then_op = std::make_shared<Add>(Xt, Zt);
    auto then_op_res = std::make_shared<Result>(then_op);

    auto Xe = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    auto Ye = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    auto Ze = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());

    auto else_op = std::make_shared<Maximum>(Xe, Xe);
    auto else_op_res = std::make_shared<Result>(else_op);
    {
        auto then_body = std::make_shared<Model>(OutputVector{then_op_res}, ParameterVector{Xt, Yt, Zt});
        auto else_body = std::make_shared<Model>(OutputVector{else_op_res}, ParameterVector{Xe, Ye, Ze});
        auto if_op = std::make_shared<If>(cond);
        if_op->set_then_body(then_body);
        if_op->set_else_body(else_body);
        if_op->set_input(X, Xt, Xe);
        if_op->set_input(Y, Yt, Ye);
        if_op->set_input(Z, Zt, Ze);
        auto res = if_op->set_output(then_op_res, else_op_res);
        function = std::make_shared<Model>(OutputVector{res}, ParameterVector{X, Y, Z});

        manager.register_pass<pass::RemoveMultiSubGraphOpDanglingParams>();
    }
    {
        auto then_body = std::make_shared<Model>(OutputVector{then_op_res}, ParameterVector{Xt, Zt});
        auto else_body = std::make_shared<Model>(OutputVector{else_op_res}, ParameterVector{Xe});
        auto if_op = std::make_shared<If>(cond);
        if_op->set_then_body(then_body);
        if_op->set_else_body(else_body);
        if_op->set_input(X, Xt, Xe);
        if_op->set_input(Z, Zt, nullptr);
        auto res = if_op->set_output(then_op_res, else_op_res);
        function_ref = std::make_shared<Model>(OutputVector{res}, ParameterVector{X, Y, Z});
    }
}

TEST_F(TransformationTestsF, RemoveIfDanglingParamFromOneBodyAndUpdateAllDescriptions) {
    std::shared_ptr<Model> f(nullptr), f_ref(nullptr);
    auto X = std::make_shared<Parameter>(element::f32, Shape{2, 4, 1});
    auto Y = std::make_shared<Parameter>(element::f32, Shape{2, 4, 1});
    auto Z = std::make_shared<Parameter>(element::f32, Shape{2, 4, 1});
    auto cond = std::make_shared<Constant>(element::boolean, Shape{1}, true);

    auto Xt = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    auto Yt = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    auto Zt = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());

    auto then_op = std::make_shared<Add>(Zt, Zt);
    auto then_op_res = std::make_shared<Result>(then_op);

    auto Xe = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    auto Ze = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());

    auto else_op = std::make_shared<Add>(std::make_shared<Maximum>(Xe, Ze), Ze);
    auto else_op_res = std::make_shared<Result>(else_op);
    {
        auto then_body = std::make_shared<Model>(OutputVector{then_op_res}, ParameterVector{Xt, Yt, Zt});
        auto else_body = std::make_shared<Model>(OutputVector{else_op_res}, ParameterVector{Xe, Ze});
        auto if_op = std::make_shared<If>(cond);
        if_op->set_then_body(then_body);
        if_op->set_else_body(else_body);
        if_op->set_input(X, Xt, Xe);
        if_op->set_input(Y, Yt, nullptr);
        if_op->set_input(Z, Zt, Ze);
        auto res = if_op->set_output(then_op_res, else_op_res);
        function = std::make_shared<Model>(OutputVector{res}, ParameterVector{X, Y, Z});

        manager.register_pass<pass::RemoveMultiSubGraphOpDanglingParams>();
    }
    {
        auto then_body = std::make_shared<Model>(OutputVector{then_op_res}, ParameterVector{Zt});
        auto else_body = std::make_shared<Model>(OutputVector{else_op_res}, ParameterVector{Xe, Ze});
        auto if_op = std::make_shared<If>(cond);
        if_op->set_then_body(then_body);
        if_op->set_else_body(else_body);
        if_op->set_input(X, nullptr, Xe);
        if_op->set_input(Z, Zt, Ze);
        auto res = if_op->set_output(then_op_res, else_op_res);
        function_ref = std::make_shared<Model>(OutputVector{res}, ParameterVector{X, Y, Z});
    }
}

TEST_F(TransformationTestsF, RemoveTensorIteratorDanglingParameter) {
    auto X = std::make_shared<Parameter>(element::f32, Shape{32, 40, 10});
    auto Y = std::make_shared<Parameter>(element::f32, Shape{32, 40, 10});
    auto M = std::make_shared<Parameter>(element::f32, Shape{32, 2, 10});

    auto Xi = std::make_shared<Parameter>(element::f32, Shape{32, 2, 10});
    auto Yi = std::make_shared<Parameter>(element::f32, Shape{32, 2, 10});
    auto M_body = std::make_shared<Parameter>(element::f32, Shape{32, 2, 10});
    auto Zo = std::make_shared<Abs>(std::make_shared<Add>(Xi, Yi));
    {
        auto body = std::make_shared<Model>(OutputVector{Zo}, ParameterVector{Xi, Yi, M_body});
        auto tensor_iterator = std::make_shared<TensorIterator>();
        tensor_iterator->set_body(body);
        tensor_iterator->set_sliced_input(Xi, X, 0, 2, 2, 39, 1);
        tensor_iterator->set_sliced_input(Yi, Y, 0, 2, 2, -1, 1);
        tensor_iterator->set_invariant_input(M_body, M);

        auto out = tensor_iterator->get_iter_value(Zo, -1);
        auto res = std::make_shared<Result>(out);
        function = std::make_shared<Model>(OutputVector{res}, ParameterVector{X, Y, M});

        manager.register_pass<pass::RemoveMultiSubGraphOpDanglingParams>();
    }
    {
        auto body = std::make_shared<Model>(OutputVector{Zo}, ParameterVector{Xi, Yi});
        auto tensor_iterator = std::make_shared<TensorIterator>();
        tensor_iterator->set_body(body);
        tensor_iterator->set_sliced_input(Xi, X, 0, 2, 2, 39, 1);
        tensor_iterator->set_sliced_input(Yi, Y, 0, 2, 2, -1, 1);

        auto out = tensor_iterator->get_iter_value(Zo, -1);
        auto res = std::make_shared<Result>(out);
        function_ref = std::make_shared<Model>(OutputVector{res}, ParameterVector{X, Y, M});
    }
}

TEST_F(TransformationTestsF, RemoveTensorIteratorManyDanglingParameters) {
    auto X = std::make_shared<Parameter>(element::f32, Shape{32, 40, 10});
    auto Y = std::make_shared<Parameter>(element::f32, Shape{32, 40, 10});
    auto Z = std::make_shared<Parameter>(element::f32, Shape{32, 40, 10});
    auto M = std::make_shared<Parameter>(element::f32, Shape{32, 2, 10});

    auto Xi = std::make_shared<Parameter>(element::f32, Shape{32, 2, 10});
    auto Yi = std::make_shared<Parameter>(element::f32, Shape{32, 2, 10});
    auto Zi = std::make_shared<Parameter>(element::f32, Shape{32, 2, 10});
    auto M_body = std::make_shared<Parameter>(element::f32, Shape{32, 2, 10});
    auto Zo = std::make_shared<Abs>(std::make_shared<Add>(Xi, Zi));
    {
        auto body = std::make_shared<Model>(OutputVector{Zo}, ParameterVector{Xi, Yi, Zi, M_body});
        auto tensor_iterator = std::make_shared<TensorIterator>();
        tensor_iterator->set_body(body);
        tensor_iterator->set_sliced_input(Xi, X, 0, 2, 2, 39, 1);
        tensor_iterator->set_sliced_input(Yi, Y, 0, 2, 2, -1, 1);
        tensor_iterator->set_sliced_input(Zi, Z, 0, 2, 2, -1, 1);
        tensor_iterator->set_invariant_input(M_body, M);

        auto out = tensor_iterator->get_iter_value(Zo, -1);
        auto res = std::make_shared<Result>(out);
        function = std::make_shared<Model>(OutputVector{res}, ParameterVector{X, Y, Z, M});

        manager.register_pass<pass::RemoveMultiSubGraphOpDanglingParams>();
    }
    {
        auto body = std::make_shared<Model>(OutputVector{Zo}, ParameterVector{Xi, Zi});
        auto tensor_iterator = std::make_shared<TensorIterator>();
        tensor_iterator->set_body(body);
        tensor_iterator->set_sliced_input(Xi, X, 0, 2, 2, 39, 1);
        tensor_iterator->set_sliced_input(Zi, Z, 0, 2, 2, -1, 1);

        auto out = tensor_iterator->get_iter_value(Zo, -1);
        auto res = std::make_shared<Result>(out);
        function_ref = std::make_shared<Model>(OutputVector{res}, ParameterVector{X, Y, Z, M});
    }
}
