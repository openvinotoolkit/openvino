// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/common_optimizations/remove_concat_zero_dim_input.hpp"
#include "transformations/common_optimizations/remove_multi_subgraph_op_dangling_params.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"

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
        model = std::make_shared<Model>(OutputVector{loop_res}, ParameterVector{a, b});

        manager.register_pass<ov::pass::RemoveMultiSubGraphOpDanglingParamsResults>();
    }
    {
        auto body = std::make_shared<Model>(OutputVector{condition, abs}, ParameterVector{bi});
        auto loop = std::make_shared<Loop>(trip_count, condition);
        loop->set_special_body_ports({-1, 0});
        loop->set_function(body);
        loop->set_invariant_input(bi, b);

        auto loop_res = std::make_shared<Result>(loop->get_iter_value(abs));
        model_ref = std::make_shared<Model>(OutputVector{loop_res}, ParameterVector{a, b});
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
        model = std::make_shared<Model>(OutputVector{loop_res}, ParameterVector{a, b, c});

        manager.register_pass<ov::pass::RemoveMultiSubGraphOpDanglingParamsResults>();
    }
    {
        auto body = std::make_shared<Model>(OutputVector{condition, abs}, ParameterVector{bi});
        auto loop = std::make_shared<Loop>(trip_count, condition);
        loop->set_special_body_ports({-1, 0});
        loop->set_function(body);
        loop->set_invariant_input(bi, b);

        auto loop_res = std::make_shared<Result>(loop->get_iter_value(abs));
        model_ref = std::make_shared<Model>(OutputVector{loop_res}, ParameterVector{a, b, c});
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
        model = std::make_shared<Model>(OutputVector{loop_res}, ParameterVector{a, b, c, d});

        manager.register_pass<ov::pass::RemoveMultiSubGraphOpDanglingParamsResults>();
    }
    {
        auto body = std::make_shared<Model>(OutputVector{condition, abs}, ParameterVector{bi, di});
        auto loop = std::make_shared<Loop>(trip_count, condition);
        loop->set_special_body_ports({-1, 0});
        loop->set_function(body);
        loop->set_invariant_input(bi, b);
        loop->set_invariant_input(di, d);

        auto loop_res = std::make_shared<Result>(loop->get_iter_value(abs));
        model_ref = std::make_shared<Model>(OutputVector{loop_res}, ParameterVector{a, b, c, d});
    }
}

TEST_F(TransformationTestsF, RemoveLoopDanglingParametersIfConcatEmptyTensor) {
    auto trip_count = std::make_shared<Constant>(element::i64, Shape{}, 10);
    auto condition = std::make_shared<Constant>(element::boolean, Shape{}, true);

    auto a = std::make_shared<Parameter>(element::f32, Shape{2, 2});
    auto ai = std::make_shared<Parameter>(element::f32, Shape{2, 2});
    auto b = std::make_shared<Parameter>(element::f32, Shape{0, 2});  // empty tensor
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
        model = std::make_shared<Model>(OutputVector{loop_res}, ParameterVector{a, b});

        manager.register_pass<ov::pass::RemoveConcatZeroDimInput>();
        manager.register_pass<ov::pass::RemoveMultiSubGraphOpDanglingParamsResults>();
    }
    {
        auto concat = std::make_shared<Concat>(NodeVector{ai}, 0);
        auto body = std::make_shared<Model>(OutputVector{condition, concat}, ParameterVector{ai});
        auto loop = std::make_shared<Loop>(trip_count, condition);
        loop->set_special_body_ports({-1, 0});
        loop->set_function(body);
        loop->set_invariant_input(ai, a);

        auto loop_res = std::make_shared<Result>(loop->get_iter_value(concat));
        model_ref = std::make_shared<Model>(OutputVector{loop_res}, ParameterVector{a, b});
    }
}

TEST_F(TransformationTestsF, RemoveIfDanglingParametersFromBodiesAndInputsConsecutive) {
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
        // if_op descriptors are [desc_0, desc_1, desc_2, desc_3]
        // desc_0, desc_2 are dangling, Parameters Y, Yte should be removed
        auto res = if_op->set_output(then_op_res, else_op_res);
        model = std::make_shared<Model>(OutputVector{res}, ParameterVector{X, Y});

        manager.register_pass<ov::pass::RemoveMultiSubGraphOpDanglingParamsResults>();
    }
    {
        auto then_body = std::make_shared<Model>(OutputVector{then_op_res}, ParameterVector{Xte});
        auto else_body = std::make_shared<Model>(OutputVector{else_op_res}, ParameterVector{Xte});
        auto if_op = std::make_shared<If>(cond);
        if_op->set_then_body(then_body);
        if_op->set_else_body(else_body);
        if_op->set_input(X, Xte, Xte);
        auto res = if_op->set_output(then_op_res, else_op_res);
        model_ref = std::make_shared<Model>(OutputVector{res}, ParameterVector{X, Y});
    }
}

TEST_F(TransformationTestsF, RemoveIfDanglingParametersFromBodiesAndInputsNotConsecutive) {
    auto X = std::make_shared<Parameter>(element::f32, Shape{2, 4, 1});
    auto Y = std::make_shared<Parameter>(element::f32, Shape{3, 4, 1});
    auto cond = std::make_shared<Constant>(element::boolean, Shape{1}, false);

    auto Xte = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    auto Yte = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());

    auto then_op = std::make_shared<Add>(Yte, Yte);
    auto then_op_res = std::make_shared<Result>(then_op);

    auto else_op = std::make_shared<Maximum>(Yte, Yte);
    auto else_op_res = std::make_shared<Result>(else_op);
    {
        auto then_body = std::make_shared<Model>(OutputVector{then_op_res}, ParameterVector{Xte, Yte});
        auto else_body = std::make_shared<Model>(OutputVector{else_op_res}, ParameterVector{Xte, Yte});
        auto if_op = std::make_shared<If>(cond);
        if_op->set_then_body(then_body);
        if_op->set_else_body(else_body);
        if_op->set_input(X, Xte, Yte);
        if_op->set_input(Y, Xte, Xte);
        // if_op descriptors are [desc_0, desc_1, desc_2, desc_3]
        // desc_0, desc_2, desc_3 are dangling, Parameters Y, Xte should be removed
        auto res = if_op->set_output(then_op_res, else_op_res);
        model = std::make_shared<Model>(OutputVector{res}, ParameterVector{X, Y});

        manager.register_pass<ov::pass::RemoveMultiSubGraphOpDanglingParamsResults>();
    }
    {
        auto then_body = std::make_shared<Model>(OutputVector{then_op_res}, ParameterVector{Yte});
        auto else_body = std::make_shared<Model>(OutputVector{else_op_res}, ParameterVector{Yte});
        auto if_op = std::make_shared<If>(cond);
        if_op->set_then_body(then_body);
        if_op->set_else_body(else_body);
        if_op->set_input(X, Yte, Yte);
        auto res = if_op->set_output(then_op_res, else_op_res);
        model_ref = std::make_shared<Model>(OutputVector{res}, ParameterVector{X, Y});
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
        model = std::make_shared<Model>(OutputVector{res}, ParameterVector{X, Y});

        manager.register_pass<ov::pass::RemoveMultiSubGraphOpDanglingParamsResults>();
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
        model_ref = std::make_shared<Model>(OutputVector{res}, ParameterVector{X, Y});
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
        model = std::make_shared<Model>(OutputVector{res}, ParameterVector{X, Y, Z});

        manager.register_pass<ov::pass::RemoveMultiSubGraphOpDanglingParamsResults>();
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
        model_ref = std::make_shared<Model>(OutputVector{res}, ParameterVector{X, Y, Z});
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
        model = std::make_shared<Model>(OutputVector{res}, ParameterVector{X, Y, Z});

        manager.register_pass<ov::pass::RemoveMultiSubGraphOpDanglingParamsResults>();
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
        model_ref = std::make_shared<Model>(OutputVector{res}, ParameterVector{X, Y, Z});
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
        model = std::make_shared<Model>(OutputVector{res}, ParameterVector{X, Y, M});

        manager.register_pass<ov::pass::RemoveMultiSubGraphOpDanglingParamsResults>();
    }
    {
        auto body = std::make_shared<Model>(OutputVector{Zo}, ParameterVector{Xi, Yi});
        auto tensor_iterator = std::make_shared<TensorIterator>();
        tensor_iterator->set_body(body);
        tensor_iterator->set_sliced_input(Xi, X, 0, 2, 2, 39, 1);
        tensor_iterator->set_sliced_input(Yi, Y, 0, 2, 2, -1, 1);

        auto out = tensor_iterator->get_iter_value(Zo, -1);
        auto res = std::make_shared<Result>(out);
        model_ref = std::make_shared<Model>(OutputVector{res}, ParameterVector{X, Y, M});
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
        model = std::make_shared<Model>(OutputVector{res}, ParameterVector{X, Y, Z, M});

        manager.register_pass<ov::pass::RemoveMultiSubGraphOpDanglingParamsResults>();
    }
    {
        auto body = std::make_shared<Model>(OutputVector{Zo}, ParameterVector{Xi, Zi});
        auto tensor_iterator = std::make_shared<TensorIterator>();
        tensor_iterator->set_body(body);
        tensor_iterator->set_sliced_input(Xi, X, 0, 2, 2, 39, 1);
        tensor_iterator->set_sliced_input(Zi, Z, 0, 2, 2, -1, 1);

        auto out = tensor_iterator->get_iter_value(Zo, -1);
        auto res = std::make_shared<Result>(out);
        model_ref = std::make_shared<Model>(OutputVector{res}, ParameterVector{X, Y, Z, M});
    }
}

TEST_F(TransformationTestsF, RemoveIfDanglingResult) {
    auto X = std::make_shared<Parameter>(element::f32, Shape{2, 4, 1});
    auto Y = std::make_shared<Parameter>(element::f32, Shape{2, 4, 1});
    auto cond = std::make_shared<Constant>(element::boolean, Shape{1}, true);

    auto Xt = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    auto Yt = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());

    auto then_op1 = std::make_shared<Add>(Xt, Yt);
    auto then_op1_res = std::make_shared<Result>(then_op1);
    auto then_op2 = std::make_shared<Subtract>(Xt, Yt);
    auto then_op2_res = std::make_shared<Result>(then_op2);

    auto Xe = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());

    auto else_op1 = std::make_shared<Add>(Xe, Xe);
    auto else_op1_res = std::make_shared<Result>(else_op1);
    auto else_op2 = std::make_shared<Subtract>(Xe, Xe);
    auto else_op2_res = std::make_shared<Result>(else_op2);
    {
        auto then_body = std::make_shared<Model>(OutputVector{then_op1_res, then_op2_res}, ParameterVector{Xt, Yt});
        auto else_body = std::make_shared<Model>(OutputVector{else_op1_res, else_op2_res}, ParameterVector{Xe});
        auto if_op = std::make_shared<If>(cond);
        if_op->set_then_body(then_body);
        if_op->set_else_body(else_body);
        if_op->set_input(X, Xt, Xe);
        if_op->set_input(Y, Yt, nullptr);
        auto res1 = if_op->set_output(then_op1_res, else_op1_res);
        auto res2 = if_op->set_output(then_op2_res, else_op2_res);
        // Not using res2 output
        model = std::make_shared<Model>(OutputVector{res1}, ParameterVector{X, Y});

        manager.register_pass<ov::pass::RemoveMultiSubGraphOpDanglingParamsResults>();
    }
    {
        auto then_body = std::make_shared<Model>(OutputVector{then_op1_res}, ParameterVector{Xt, Yt});
        auto else_body = std::make_shared<Model>(OutputVector{else_op1_res}, ParameterVector{Xe});
        auto if_op = std::make_shared<If>(cond);
        if_op->set_then_body(then_body);
        if_op->set_else_body(else_body);
        if_op->set_input(X, Xt, Xe);
        if_op->set_input(Y, Yt, nullptr);
        auto res1 = if_op->set_output(then_op1_res, else_op1_res);
        model_ref = std::make_shared<Model>(OutputVector{res1}, ParameterVector{X, Y});
    }
}

TEST_F(TransformationTestsF, RemoveLoopDanglingResults) {
    auto trip_count = std::make_shared<Constant>(element::i64, Shape{}, 10);
    auto condition = std::make_shared<Constant>(element::boolean, Shape{}, true);

    auto a = std::make_shared<Parameter>(element::f32, Shape{2, 2});
    auto ai = std::make_shared<Parameter>(element::f32, Shape{2, 2});
    auto b = std::make_shared<Parameter>(element::f32, Shape{2, 2});
    auto bi = std::make_shared<Parameter>(element::f32, Shape{2, 2});

    auto mul = std::make_shared<Multiply>(ai, bi);
    auto abs1 = std::make_shared<Abs>(mul);
    auto add = std::make_shared<Add>(ai, bi);
    auto abs2 = std::make_shared<Abs>(add);
    {
        auto body = std::make_shared<Model>(OutputVector{condition, abs1, abs2}, ParameterVector{ai, bi});
        auto loop = std::make_shared<Loop>(trip_count, condition);
        loop->set_special_body_ports({-1, 0});
        loop->set_function(body);
        loop->set_invariant_input(ai, a);
        loop->set_invariant_input(bi, b);

        auto loop_res = std::make_shared<Result>(loop->get_iter_value(abs1));
        loop->get_iter_value(abs2);
        // abs2 result is unused
        model = std::make_shared<Model>(OutputVector{loop_res}, ParameterVector{a, b});

        manager.register_pass<ov::pass::RemoveMultiSubGraphOpDanglingParamsResults>();
    }
    {
        auto body = std::make_shared<Model>(OutputVector{condition, abs1}, ParameterVector{ai, bi});
        auto loop = std::make_shared<Loop>(trip_count, condition);
        loop->set_special_body_ports({-1, 0});
        loop->set_function(body);
        loop->set_invariant_input(ai, a);
        loop->set_invariant_input(bi, b);

        auto loop_res = std::make_shared<Result>(loop->get_iter_value(abs1));
        model_ref = std::make_shared<Model>(OutputVector{loop_res}, ParameterVector{a, b});
    }
}

TEST_F(TransformationTestsF, RemoveLoopDanglingParamsAndResults) {
    auto trip_count = std::make_shared<Constant>(element::i64, Shape{}, 10);
    auto condition = std::make_shared<Constant>(element::boolean, Shape{}, true);

    auto a = std::make_shared<Parameter>(element::f32, Shape{2, 2});
    auto ai = std::make_shared<Parameter>(element::f32, Shape{2, 2});
    auto b = std::make_shared<Parameter>(element::f32, Shape{2, 2});
    auto bi = std::make_shared<Parameter>(element::f32, Shape{2, 2});
    auto c = std::make_shared<Parameter>(element::f32, Shape{2, 2});
    auto ci = std::make_shared<Parameter>(element::f32, Shape{2, 2});
    auto d = std::make_shared<Parameter>(element::f32, Shape{2, 2});

    auto mul = std::make_shared<Multiply>(ai, ai);
    auto abs1 = std::make_shared<Abs>(mul);
    auto add = std::make_shared<Add>(bi, bi);
    auto abs2 = std::make_shared<Abs>(add);
    {
        auto body = std::make_shared<Model>(OutputVector{condition, abs1, abs2}, ParameterVector{ai, bi, ci});
        auto loop = std::make_shared<Loop>(trip_count, condition);
        loop->set_special_body_ports({-1, 0});
        loop->set_function(body);
        loop->set_invariant_input(ai, a);
        loop->set_invariant_input(ci, d);
        loop->set_invariant_input(bi, b);
        loop->set_invariant_input(ci, c);

        auto loop_res = std::make_shared<Result>(loop->get_iter_value(abs1));
        loop->get_iter_value(abs2);
        // abs2 result is unused
        model = std::make_shared<Model>(OutputVector{loop_res}, ParameterVector{a, b, c, d});

        manager.register_pass<ov::pass::RemoveMultiSubGraphOpDanglingParamsResults>();
    }
    {
        auto body = std::make_shared<Model>(OutputVector{condition, abs1}, ParameterVector{ai});
        auto loop = std::make_shared<Loop>(trip_count, condition);
        loop->set_special_body_ports({-1, 0});
        loop->set_function(body);
        loop->set_invariant_input(ai, a);

        auto loop_res = std::make_shared<Result>(loop->get_iter_value(abs1));
        model_ref = std::make_shared<Model>(OutputVector{loop_res}, ParameterVector{a, b});
    }
}

TEST_F(TransformationTestsF, RemoveLoopMultipleDanglingResults) {
    auto trip_count = std::make_shared<Constant>(element::i64, Shape{}, 10);
    auto condition = std::make_shared<Constant>(element::boolean, Shape{}, true);

    auto a = std::make_shared<Parameter>(element::f32, Shape{2, 2});
    auto ai = std::make_shared<Parameter>(element::f32, Shape{2, 2});
    auto b = std::make_shared<Parameter>(element::f32, Shape{2, 2});
    auto bi = std::make_shared<Parameter>(element::f32, Shape{2, 2});

    auto mul = std::make_shared<Multiply>(ai, bi);
    auto abs1 = std::make_shared<Abs>(mul);
    auto add = std::make_shared<Add>(ai, bi);
    auto abs2 = std::make_shared<Abs>(add);
    auto sub = std::make_shared<Subtract>(ai, bi);
    auto abs3 = std::make_shared<Abs>(sub);
    auto div = std::make_shared<Divide>(ai, bi);
    auto abs4 = std::make_shared<Abs>(div);
    {
        auto body = std::make_shared<Model>(OutputVector{condition, abs1, abs2, abs3, abs4}, ParameterVector{ai, bi});
        auto loop = std::make_shared<Loop>(trip_count, condition);
        loop->set_special_body_ports({-1, 0});
        loop->set_function(body);
        loop->set_invariant_input(ai, a);
        loop->set_invariant_input(bi, b);

        auto loop_res = std::make_shared<Result>(loop->get_iter_value(abs1));
        loop->get_iter_value(abs2);
        auto loop_res2 = std::make_shared<Result>(loop->get_iter_value(abs3));
        loop->get_iter_value(abs4);
        // abs2 and abs4 result is unused
        model = std::make_shared<Model>(OutputVector{loop_res, loop_res2}, ParameterVector{a, b});

        manager.register_pass<ov::pass::RemoveMultiSubGraphOpDanglingParamsResults>();
    }
    {
        auto body = std::make_shared<Model>(OutputVector{condition, abs1, abs3}, ParameterVector{ai, bi});
        auto loop = std::make_shared<Loop>(trip_count, condition);
        loop->set_special_body_ports({-1, 0});
        loop->set_function(body);
        loop->set_invariant_input(ai, a);
        loop->set_invariant_input(bi, b);

        auto loop_res = std::make_shared<Result>(loop->get_iter_value(abs1));
        auto loop_res2 = std::make_shared<Result>(loop->get_iter_value(abs3));
        model_ref = std::make_shared<Model>(OutputVector{loop_res, loop_res2}, ParameterVector{a, b});
    }
}

TEST_F(TransformationTestsF, RemoveLoopDanglingResultsSpecialOutPortMoved) {
    auto trip_count = std::make_shared<Constant>(element::i64, Shape{}, 10);
    auto condition = std::make_shared<Constant>(element::boolean, Shape{}, true);

    auto a = std::make_shared<Parameter>(element::f32, Shape{2, 2});
    auto ai = std::make_shared<Parameter>(element::f32, Shape{2, 2});
    auto b = std::make_shared<Parameter>(element::f32, Shape{2, 2});
    auto bi = std::make_shared<Parameter>(element::f32, Shape{2, 2});

    auto mul = std::make_shared<Multiply>(ai, bi);
    auto abs1 = std::make_shared<Abs>(mul);
    auto add = std::make_shared<Add>(ai, bi);
    auto abs2 = std::make_shared<Abs>(add);
    {
        auto body = std::make_shared<Model>(OutputVector{abs1, abs2, condition}, ParameterVector{ai, bi});
        auto loop = std::make_shared<Loop>(trip_count, condition);
        loop->set_special_body_ports({-1, 2});
        loop->set_function(body);
        loop->set_invariant_input(ai, a);
        loop->set_invariant_input(bi, b);

        auto loop_res = std::make_shared<Result>(loop->get_iter_value(abs1));
        loop->get_iter_value(abs2);
        // abs2 result is unused
        model = std::make_shared<Model>(OutputVector{loop_res}, ParameterVector{a, b});

        manager.register_pass<ov::pass::RemoveMultiSubGraphOpDanglingParamsResults>();
    }
    {
        auto body = std::make_shared<Model>(OutputVector{abs1, condition}, ParameterVector{ai, bi});
        auto loop = std::make_shared<Loop>(trip_count, condition);
        loop->set_special_body_ports({-1, 1});
        loop->set_function(body);
        loop->set_invariant_input(ai, a);
        loop->set_invariant_input(bi, b);

        auto loop_res = std::make_shared<Result>(loop->get_iter_value(abs1));
        model_ref = std::make_shared<Model>(OutputVector{loop_res}, ParameterVector{a, b});
    }
}

TEST_F(TransformationTestsF, RemoveTensorIteratorDanglingResult) {
    auto X = std::make_shared<Parameter>(element::f32, Shape{32, 40, 10});
    auto Y = std::make_shared<Parameter>(element::f32, Shape{32, 40, 10});

    auto Xi = std::make_shared<Parameter>(element::f32, Shape{32, 2, 10});
    auto Yi = std::make_shared<Parameter>(element::f32, Shape{32, 2, 10});
    auto Zo = std::make_shared<Abs>(std::make_shared<Add>(Xi, Yi));
    auto Zo2 = std::make_shared<Abs>(std::make_shared<Subtract>(Xi, Yi));
    {
        auto body = std::make_shared<Model>(OutputVector{Zo, Zo2}, ParameterVector{Xi, Yi});
        auto tensor_iterator = std::make_shared<TensorIterator>();
        tensor_iterator->set_body(body);
        tensor_iterator->set_sliced_input(Xi, X, 0, 2, 2, 39, 1);
        tensor_iterator->set_sliced_input(Yi, Y, 0, 2, 2, -1, 1);

        auto out = tensor_iterator->get_iter_value(Zo, -1);
        auto out2 = tensor_iterator->get_iter_value(Zo2, -1);
        auto res = std::make_shared<Result>(out);
        // out2 is not used
        model = std::make_shared<Model>(OutputVector{res}, ParameterVector{X, Y});

        manager.register_pass<ov::pass::RemoveMultiSubGraphOpDanglingParamsResults>();
    }
    {
        auto body = std::make_shared<Model>(OutputVector{Zo}, ParameterVector{Xi, Yi});
        auto tensor_iterator = std::make_shared<TensorIterator>();
        tensor_iterator->set_body(body);
        tensor_iterator->set_sliced_input(Xi, X, 0, 2, 2, 39, 1);
        tensor_iterator->set_sliced_input(Yi, Y, 0, 2, 2, -1, 1);

        auto out = tensor_iterator->get_iter_value(Zo, -1);
        auto res = std::make_shared<Result>(out);
        model_ref = std::make_shared<Model>(OutputVector{res}, ParameterVector{X, Y});
    }
}

TEST_F(TransformationTestsF, RemoveTensorIteratorMultipleDanglingResult) {
    auto X = std::make_shared<Parameter>(element::f32, Shape{32, 40, 10});
    auto Y = std::make_shared<Parameter>(element::f32, Shape{32, 40, 10});

    auto Xi = std::make_shared<Parameter>(element::f32, Shape{32, 2, 10});
    auto Yi = std::make_shared<Parameter>(element::f32, Shape{32, 2, 10});
    auto Zo1 = std::make_shared<Abs>(std::make_shared<Add>(Xi, Yi));
    auto Zo2 = std::make_shared<Abs>(std::make_shared<Subtract>(Xi, Yi));
    auto Zo3 = std::make_shared<Abs>(std::make_shared<Multiply>(Xi, Yi));
    auto Zo4 = std::make_shared<Abs>(std::make_shared<Divide>(Xi, Yi));
    {
        auto body = std::make_shared<Model>(OutputVector{Zo1, Zo2, Zo3, Zo4}, ParameterVector{Xi, Yi});
        auto tensor_iterator = std::make_shared<TensorIterator>();
        tensor_iterator->set_body(body);
        tensor_iterator->set_sliced_input(Xi, X, 0, 2, 2, 39, 1);
        tensor_iterator->set_sliced_input(Yi, Y, 0, 2, 2, -1, 1);

        auto out1 = tensor_iterator->get_iter_value(Zo1, -1);
        auto out2 = tensor_iterator->get_iter_value(Zo2, -1);
        auto out3 = tensor_iterator->get_iter_value(Zo3, -1);
        auto out4 = tensor_iterator->get_iter_value(Zo4, -1);
        // out1 and out3 is not used
        model = std::make_shared<Model>(OutputVector{out2, out4}, ParameterVector{X, Y});

        manager.register_pass<ov::pass::RemoveMultiSubGraphOpDanglingParamsResults>();
    }
    {
        auto body = std::make_shared<Model>(OutputVector{Zo2, Zo4}, ParameterVector{Xi, Yi});
        auto tensor_iterator = std::make_shared<TensorIterator>();
        tensor_iterator->set_body(body);
        tensor_iterator->set_sliced_input(Xi, X, 0, 2, 2, 39, 1);
        tensor_iterator->set_sliced_input(Yi, Y, 0, 2, 2, -1, 1);

        auto out2 = tensor_iterator->get_iter_value(Zo2, -1);
        auto out4 = tensor_iterator->get_iter_value(Zo4, -1);
        model_ref = std::make_shared<Model>(OutputVector{out2, out4}, ParameterVector{X, Y});
    }
}

TEST_F(TransformationTestsF, RemoveLoopDanglingResultsPreserveMerged) {
    auto trip_count = std::make_shared<Constant>(element::i64, Shape{}, 10);
    auto condition = std::make_shared<Constant>(element::boolean, Shape{}, true);

    auto a = std::make_shared<Parameter>(element::f32, Shape{2, 2});
    auto ai = std::make_shared<Parameter>(element::f32, Shape{2, 2});
    auto b = std::make_shared<Parameter>(element::f32, Shape{2, 2});
    auto bi = std::make_shared<Parameter>(element::f32, Shape{2, 2});

    auto mul = std::make_shared<Multiply>(ai, bi);
    auto abs1 = std::make_shared<Abs>(mul);
    auto add = std::make_shared<Add>(ai, bi);
    auto abs2 = std::make_shared<Abs>(add);
    auto sub = std::make_shared<Subtract>(ai, bi);
    auto abs3 = std::make_shared<Abs>(sub);
    {
        auto body = std::make_shared<Model>(OutputVector{condition, abs1, abs2, abs3}, ParameterVector{ai, bi});
        auto loop = std::make_shared<Loop>(trip_count, condition);
        loop->set_special_body_ports({-1, 0});
        loop->set_function(body);
        loop->set_invariant_input(ai, a);
        loop->set_merged_input(bi, b, abs3);

        auto loop_res = std::make_shared<Result>(loop->get_iter_value(abs1));
        loop->get_iter_value(abs2);
        // abs2 result is unused
        model = std::make_shared<Model>(OutputVector{loop_res}, ParameterVector{a, b});

        manager.register_pass<ov::pass::RemoveMultiSubGraphOpDanglingParamsResults>();
    }
    {
        auto body = std::make_shared<Model>(OutputVector{condition, abs1, abs3}, ParameterVector{ai, bi});
        auto loop = std::make_shared<Loop>(trip_count, condition);
        loop->set_special_body_ports({-1, 0});
        loop->set_function(body);
        loop->set_invariant_input(ai, a);
        loop->set_merged_input(bi, b, abs3);

        auto loop_res = std::make_shared<Result>(loop->get_iter_value(abs1));
        model_ref = std::make_shared<Model>(OutputVector{loop_res}, ParameterVector{a, b});
    }
}
