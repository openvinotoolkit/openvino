// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>
#include <queue>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset7.hpp>
#include <transformations/preprocessing/scale_inputs.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include <ngraph/pass/manager.hpp>
#include <ngraph/pass/visualize_tree.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

TEST(TransformationTests, ScaleInputs_float) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto data1 = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32, ngraph::Shape{3, 1, 2});
        data1->set_friendly_name("input1");
        auto data2 = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32, ngraph::Shape{3, 1, 2});
        data2->set_friendly_name("input2");
        auto add = std::make_shared<ngraph::opset7::Add>(data1, data2);
        add->set_friendly_name("add");
        auto res = std::make_shared<ngraph::opset7::Result>(add);
        res->set_friendly_name("Result");

        f = std::make_shared<ngraph::Function>(ngraph::ResultVector{res}, ngraph::ParameterVector{data1, data2});

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<ngraph::pass::ScaleInputs>(0.5);
        m.run_passes(f);
    }

    {
        auto data1 = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32, ngraph::Shape{3, 1, 2});
        data1->set_friendly_name("input1");
        auto mul_const1 = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {0.5});
        mul_const1->set_friendly_name("input1/scale/Fused_Mul_Factor");
        auto mul1 = std::make_shared<ngraph::opset7::Multiply>(data1, mul_const1);
        mul1->set_friendly_name("input1/scale/Fused_Mul");
        auto data2 = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32, ngraph::Shape{3, 1, 2});
        data2->set_friendly_name("input2");
        auto mul_const2 = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {0.5});
        mul_const2->set_friendly_name("input2/scale/Fused_Mul_Factor");
        auto mul2 = std::make_shared<ngraph::opset7::Multiply>(data2, mul_const2);
        mul2->set_friendly_name("input2/scale/Fused_Mul");
        auto add = std::make_shared<ngraph::opset7::Add>(mul1, mul2);
        add->set_friendly_name("add");
        auto res = std::make_shared<ngraph::opset7::Result>(add);
        res->set_friendly_name("Result");

        f_ref = std::make_shared<ngraph::Function>(ngraph::ResultVector{res}, ngraph::ParameterVector{data1, data2});
    }

    const FunctionsComparator func_comparator =
            FunctionsComparator::with_default().enable(FunctionsComparator::NAMES_ALL);
    const FunctionsComparator::Result res = func_comparator(f, f_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

TEST(TransformationTests, ScaleInputs_map) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto data1 = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 5, 5});
        data1->set_friendly_name("input1");
        auto data2 = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 5, 5});
        data2->set_friendly_name("input2");
        auto add = std::make_shared<ngraph::opset7::Add>(data1, data2);
        add->set_friendly_name("add");
        auto res = std::make_shared<ngraph::opset7::Result>(add);
        res->set_friendly_name("Result");

        f = std::make_shared<ngraph::Function>(ngraph::ResultVector{res}, ngraph::ParameterVector{data1, data2});

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        std::map<std::string, std::vector<float>> map;
        map.insert({"input1", {0.1, 0.2, 0.3}});
        m.register_pass<ngraph::pass::ScaleInputs>(map);
        m.run_passes(f);
    }

    {
        auto data1 = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 5, 5});
        data1->set_friendly_name("input1");
        auto mul_const1 = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1, 3, 1, 1}, {0.1, 0.2, 0.3});
        mul_const1->set_friendly_name("input1/scale/Fused_Mul_Factor");
        auto mul1 = std::make_shared<ngraph::opset7::Multiply>(data1, mul_const1);
        mul1->set_friendly_name("input1/scale/Fused_Mul");
        auto data2 = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 5, 5});
        data2->set_friendly_name("input2");
        auto add = std::make_shared<ngraph::opset7::Add>(mul1, data2);
        add->set_friendly_name("add");
        auto res = std::make_shared<ngraph::opset7::Result>(add);
        res->set_friendly_name("Result");

        f_ref = std::make_shared<ngraph::Function>(ngraph::ResultVector{res}, ngraph::ParameterVector{data1, data2});
    }

    const FunctionsComparator func_comparator =
            FunctionsComparator::with_default().enable(FunctionsComparator::NAMES_ALL);
    const FunctionsComparator::Result res = func_comparator(f, f_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

//TEST(TransformationTests, ConvertDivideNegative) {
//    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
//    {
//        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::i32, ngraph::Shape{3, 1, 2});
//        auto divide_constant = ngraph::opset1::Constant::create(ngraph::element::i32, ngraph::Shape{1}, {2});
//        auto divide = std::make_shared<ngraph::opset1::Divide>(data, divide_constant);
//
//        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{divide}, ngraph::ParameterVector{data});
//
//        ngraph::pass::Manager m;
//        m.register_pass<ngraph::pass::InitNodeInfo>();
//        m.register_pass<ngraph::pass::ConvertDivide>();
//        m.run_passes(f);
//        ASSERT_NO_THROW(check_rt_info(f));
//    }
//
//    {
//        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::i32, ngraph::Shape{3, 1, 2});
//        auto divide_constant = ngraph::opset1::Constant::create(ngraph::element::i32, ngraph::Shape{1}, {2});
//        auto divide = std::make_shared<ngraph::opset1::Divide>(data, divide_constant);
//
//        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{divide}, ngraph::ParameterVector{data});
//    }
//
//    auto res = compare_functions(f, f_ref);
//    ASSERT_TRUE(res.first) << res.second;
//}
//
//TEST(TransformationTests, ConvertDivideScalar) {
//    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
//    {
//        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{});
//        auto divide_constant = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{}, {1.5});
//        auto divide = std::make_shared<ngraph::opset1::Divide>(data, divide_constant);
//
//        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{divide}, ngraph::ParameterVector{data});
//
//        NGRAPH_CHECK(divide->get_output_partial_shape(0).rank().get_length() == 0);
//
//        ngraph::pass::Manager m;
//        m.register_pass<ngraph::pass::InitNodeInfo>();
//        m.register_pass<ngraph::pass::ConvertDivide>();
//        m.run_passes(f);
//        ASSERT_NO_THROW(check_rt_info(f));
//    }
//
//    {
//        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{});
//        auto divide_constant = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{}, {1.5});
//        auto pow = std::make_shared<ngraph::opset1::Power>(divide_constant,
//                                                           ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{}, {-1}));
//        auto mul = std::make_shared<ngraph::opset1::Multiply>(data, pow);
//
//        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{mul}, ngraph::ParameterVector{data});
//
//        NGRAPH_CHECK(mul->get_output_partial_shape(0).rank().get_length() == 0);
//    }
//
//    auto res = compare_functions(f, f_ref);
//    ASSERT_TRUE(res.first) << res.second;
//}
