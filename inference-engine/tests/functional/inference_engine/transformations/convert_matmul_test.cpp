// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_common.hpp"

#include <string>
#include <sstream>
#include <fstream>
#include <memory>
#include <queue>
#include <map>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <ngraph_ops/fully_connected.hpp>
#include <transformations/convert_opset1_to_legacy/convert_matmul_to_fc_or_gemm.hpp>
#include <transformations/convert_opset1_to_legacy/reshape_fully_connected.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>

#include "ngraph_test_utils.hpp"

using namespace testing;

TEST(TransformationTests, ConvertMatMulTest1) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 1, 2});
        auto input2 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{2, 1});
        auto matmul = std::make_shared<ngraph::opset1::MatMul>(input1, input2, false, false);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{input1, input2});

        ngraph::pass::InitNodeInfo().run_on_function(f);
        ngraph::pass::ConvertMatMulToFCorGemm().run_on_function(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 1, 2});
        auto input2 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{2, 1});

        auto reshape = ngraph::op::util::reshapeTo(input2, {1, 2, 1});

        auto matmul = std::make_shared<ngraph::opset1::MatMul>(input1, reshape, false, false);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{input1, input2});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertMatMulTest2) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 1, 2});
        auto input2 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{2});
        auto matmul = std::make_shared<ngraph::opset1::MatMul>(input1, input2, false, false);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{input1, input2});

        ngraph::pass::InitNodeInfo().run_on_function(f);
        ngraph::pass::ConvertMatMulToFCorGemm().run_on_function(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 1, 2});
        auto input2 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{2});

        auto reshape = ngraph::op::util::reshapeTo(input2, {1, 2, 1});
        auto matmul = std::make_shared<ngraph::opset1::MatMul>(input1, reshape, false, false);
        auto reshape_output = ngraph::op::util::reshapeTo(matmul, {3, 1});

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{reshape_output}, ngraph::ParameterVector{input1, input2});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertMatMulTest3) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{2});
        auto input2 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 2, 1});
        auto matmul = std::make_shared<ngraph::opset1::MatMul>(input1, input2, false, false);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{input1, input2});
        ngraph::pass::InitNodeInfo().run_on_function(f);
        ngraph::pass::ConvertMatMulToFCorGemm().run_on_function(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{2});
        auto input2 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 2, 1});

        auto reshape = ngraph::op::util::reshapeTo(input1, {1, 1, 2});
        auto matmul = std::make_shared<ngraph::opset1::MatMul>(reshape, input2, false, false);
        auto reshape_output = ngraph::op::util::reshapeTo(matmul, {3, 1});

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{reshape_output}, ngraph::ParameterVector{input1, input2});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertMatMulTest4) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 1, 2});
        auto input2 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 2, 1});
        auto matmul = std::make_shared<ngraph::opset1::MatMul>(input1, input2, false, false);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{input1, input2});
        ngraph::pass::InitNodeInfo().run_on_function(f);
        ngraph::pass::ConvertMatMulToFCorGemm().run_on_function(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 1, 2});
        auto input2 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 2, 1});
        auto matmul = std::make_shared<ngraph::opset1::MatMul>(input1, input2, false, false);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{input1, input2});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertMatMulTest5) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 2, 2});
        auto input2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{2, 2}, {1});
        auto matmul = std::make_shared<ngraph::opset1::MatMul>(input1, input2, false, true);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{input1});
        ngraph::pass::InitNodeInfo().run_on_function(f);
        ngraph::pass::ConvertMatMulToFCorGemm().run_on_function(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 2, 2});
        auto input2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{2, 2}, {1});
        auto input3 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{2}, {1});
        auto matmul = std::make_shared<ngraph::op::FullyConnected>(input1, input2, input3, ngraph::Shape{3, 2, 2});

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{input1});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertMatMulTest6) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 2, 2});
        auto input2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{2, 2}, {1});
        auto matmul = std::make_shared<ngraph::opset1::MatMul>(input1, input2, false, true);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{input1});
        ngraph::pass::ConvertMatMulToFCorGemm().run_on_function(f);
        ngraph::pass::ReshapeFullyConnected().run_on_function(f);
    }

    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 2, 2});
        auto input2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{2, 2}, {1});
        auto input3 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{2}, {1});
        auto reshape_begin = ngraph::op::util::reshapeTo(input1, ngraph::Shape{6, 2});
        auto fc = std::make_shared<ngraph::op::FullyConnected>(reshape_begin, input2, input3, ngraph::Shape{6, 2});
        auto reshape_end = ngraph::op::util::reshapeTo(fc, ngraph::Shape{3, 2, 2});

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{reshape_end}, ngraph::ParameterVector{input1});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertMatMulTest7) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 2, 2});
        auto input2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{2, 2}, {1});
        auto matmul = std::make_shared<ngraph::opset1::MatMul>(input1, input2, false, true);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{input1});

        ngraph::pass::InitNodeInfo().run_on_function(f);
        ngraph::pass::ConvertMatMulToFCorGemm().run_on_function(f);

        auto callback = [](const std::shared_ptr<const ngraph::Node> & node) -> bool {
            if (auto fc_op = std::dynamic_pointer_cast<const ngraph::op::FullyConnected>(node)) {
                if (fc_op->input_value(0).get_shape().size() == 3) {
                    return true;
                }
            }
            return false;
        };
        auto p = ngraph::pass::ReshapeFullyConnected();
        p.set_callback(callback);
        p.run_on_function(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 2, 2});
        auto input2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{2, 2}, {1});
        auto input3 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{2}, {1});
        auto matmul = std::make_shared<ngraph::op::FullyConnected>(input1, input2, input3, ngraph::Shape{3, 2, 2});

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{matmul}, ngraph::ParameterVector{input1});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}