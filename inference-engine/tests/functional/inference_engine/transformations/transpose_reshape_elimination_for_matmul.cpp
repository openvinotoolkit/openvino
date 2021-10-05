// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <queue>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <transformations/common_optimizations/transpose_reshape_elimination_for_matmul.hpp>
#include <transformations/init_node_info.hpp>
#include <ngraph/pass/manager.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"


using namespace testing;
using namespace ngraph;


TEST(TransformationTests, TransposeReshapeEliminationForMatMul) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);

    Shape data_shape_1{10, 2};
    Shape data_shape_2{10, 2, 25};
    {
        auto data_1 = std::make_shared<opset1::Parameter>(element::f32, data_shape_1);
        auto data_2 = std::make_shared<opset1::Parameter>(element::f32, data_shape_2);
        auto const_transpose_before = opset1::Constant::create(element::i32, Shape{3}, {1, 2, 0});
        auto transpose_before = std::make_shared<opset1::Transpose>(data_2, const_transpose_before);
        auto const_reshape_before = opset1::Constant::create(element::i32, Shape{2}, {2, 250});
        auto reshape_before = std::make_shared<opset1::Reshape>(transpose_before, const_reshape_before, false);
        auto matmul = std::make_shared<opset1::MatMul>(data_1, reshape_before);
        auto const_reshape_after = opset1::Constant::create(element::i32, Shape{3}, {10, 10, 25});
        auto reshape_after = std::make_shared<opset1::Reshape>(matmul, const_reshape_after, false);
        auto const_tranpose_after = opset1::Constant::create(element::i32, Shape{3}, {2, 0, 1});
        auto tranpose_after = std::make_shared<opset1::Transpose>(reshape_after, const_tranpose_after);
        f = std::make_shared<Function>(NodeVector{tranpose_after}, ParameterVector{data_1, data_2});
        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::TransposeReshapeEliminationForMatmul>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }
    {
        auto data_1 = std::make_shared<opset1::Parameter>(element::f32, data_shape_1);
        auto data_2 = std::make_shared<opset1::Parameter>(element::f32, data_shape_2);
        auto matmul = std::make_shared<opset1::MatMul>(data_1, data_2);
        f_ref = std::make_shared<Function>(NodeVector{matmul}, ParameterVector{data_1, data_2});
    }

    auto res = compare_functions(f, f_ref, true);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, TransposeReshapeEliminationForMatMul_TransposedA) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);

    Shape data_shape_1{2, 10};
    Shape data_shape_2{10, 2, 25};
    {
        auto data_1 = std::make_shared<opset1::Parameter>(element::f32, data_shape_1);
        auto data_2 = std::make_shared<opset1::Parameter>(element::f32, data_shape_2);
        auto const_transpose_before = opset1::Constant::create(element::i32, Shape{3}, {1, 2, 0});
        auto transpose_before = std::make_shared<opset1::Transpose>(data_2, const_transpose_before);
        auto const_reshape_before = opset1::Constant::create(element::i32, Shape{2}, {2, 250});
        auto reshape_before = std::make_shared<opset1::Reshape>(transpose_before, const_reshape_before, false);
        auto matmul = std::make_shared<opset1::MatMul>(data_1, reshape_before, true, false);
        auto const_reshape_after = opset1::Constant::create(element::i32, Shape{3}, {10, 10, 25});
        auto reshape_after = std::make_shared<opset1::Reshape>(matmul, const_reshape_after, false);
        auto const_tranpose_after = opset1::Constant::create(element::i32, Shape{3}, {2, 0, 1});
        auto tranpose_after = std::make_shared<opset1::Transpose>(reshape_after, const_tranpose_after);
        f = std::make_shared<Function>(NodeVector{tranpose_after}, ParameterVector{data_1, data_2});
        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::TransposeReshapeEliminationForMatmul>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }
    {
        auto data_1 = std::make_shared<opset1::Parameter>(element::f32, data_shape_1);
        auto data_2 = std::make_shared<opset1::Parameter>(element::f32, data_shape_2);
        auto matmul = std::make_shared<opset1::MatMul>(data_1, data_2, true, false);
        f_ref = std::make_shared<Function>(NodeVector{matmul}, ParameterVector{data_1, data_2});
    }

    auto res = compare_functions(f, f_ref, true);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, TransposeReshapeEliminationForMatMul_TransposedB) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);

    Shape data_shape_1{10, 2};
    Shape data_shape_2{10, 2, 25};
    {
        auto data_1 = std::make_shared<opset1::Parameter>(element::f32, data_shape_1);
        auto data_2 = std::make_shared<opset1::Parameter>(element::f32, data_shape_2);
        auto const_transpose_before = opset1::Constant::create(element::i32, Shape{3}, {0, 2, 1});
        auto transpose_before = std::make_shared<opset1::Transpose>(data_2, const_transpose_before);
        auto const_reshape_before = opset1::Constant::create(element::i32, Shape{2}, {250, 2});
        auto reshape_before = std::make_shared<opset1::Reshape>(transpose_before, const_reshape_before, false);
        auto matmul = std::make_shared<opset1::MatMul>(data_1, reshape_before, false, true);
        auto const_reshape_after = opset1::Constant::create(element::i32, Shape{3}, {10, 10, 25});
        auto reshape_after = std::make_shared<opset1::Reshape>(matmul, const_reshape_after, false);
        auto const_tranpose_after = opset1::Constant::create(element::i32, Shape{3}, {1, 0, 2});
        auto tranpose_after = std::make_shared<opset1::Transpose>(reshape_after, const_tranpose_after);
        f = std::make_shared<Function>(NodeVector{tranpose_after}, ParameterVector{data_1, data_2});
        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::TransposeReshapeEliminationForMatmul>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }
    {
        auto data_1 = std::make_shared<opset1::Parameter>(element::f32, data_shape_1);
        auto data_2 = std::make_shared<opset1::Parameter>(element::f32, data_shape_2);
        auto matmul = std::make_shared<opset1::MatMul>(data_1, data_2);
        f_ref = std::make_shared<Function>(NodeVector{matmul}, ParameterVector{data_1, data_2});
    }

    auto res = compare_functions(f, f_ref, true);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, TransposeReshapeEliminationForMatMul_TransposedAB) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);

    Shape data_shape_1{2, 10};
    Shape data_shape_2{10, 2, 25};
    {
        auto data_1 = std::make_shared<opset1::Parameter>(element::f32, data_shape_1);
        auto data_2 = std::make_shared<opset1::Parameter>(element::f32, data_shape_2);
        auto const_transpose_before = opset1::Constant::create(element::i32, Shape{3}, {0, 2, 1});
        auto transpose_before = std::make_shared<opset1::Transpose>(data_2, const_transpose_before);
        auto const_reshape_before = opset1::Constant::create(element::i32, Shape{2}, {250, 2});
        auto reshape_before = std::make_shared<opset1::Reshape>(transpose_before, const_reshape_before, false);
        auto matmul = std::make_shared<opset1::MatMul>(data_1, reshape_before, true, true);
        auto const_reshape_after = opset1::Constant::create(element::i32, Shape{3}, {10, 10, 25});
        auto reshape_after = std::make_shared<opset1::Reshape>(matmul, const_reshape_after, false);
        auto const_tranpose_after = opset1::Constant::create(element::i32, Shape{3}, {1, 0, 2});
        auto tranpose_after = std::make_shared<opset1::Transpose>(reshape_after, const_tranpose_after);
        f = std::make_shared<Function>(NodeVector{tranpose_after}, ParameterVector{data_1, data_2});
        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::TransposeReshapeEliminationForMatmul>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }
    {
        auto data_1 = std::make_shared<opset1::Parameter>(element::f32, data_shape_1);
        auto data_2 = std::make_shared<opset1::Parameter>(element::f32, data_shape_2);
        auto matmul = std::make_shared<opset1::MatMul>(data_1, data_2, true, false);
        f_ref = std::make_shared<Function>(NodeVector{matmul}, ParameterVector{data_1, data_2});
    }

    auto res = compare_functions(f, f_ref, true);
    ASSERT_TRUE(res.first) << res.second;
}
