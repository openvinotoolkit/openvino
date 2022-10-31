// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>
#include <queue>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <transformations/op_conversions/convert_convertlike.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include <ngraph/pass/manager.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

TEST(TransformationTests, ConvertConvertLike) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto data = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{3, 1, 2});
        auto like = ngraph::opset8::Constant::create(ngraph::element::i32, ngraph::Shape{1}, {1});
        auto cvtlike = std::make_shared<ngraph::opset8::ConvertLike>(data, like);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{cvtlike}, ngraph::ParameterVector{data});

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<ngraph::pass::ConvertConvertLike>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto data = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{3, 1, 2});
        auto cvt = std::make_shared<ngraph::opset8::Convert>(data, ngraph::element::i32);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{cvt}, ngraph::ParameterVector{data});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertConvertLike2) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto data = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{3, 1, 2});
        auto data2 = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i8, ngraph::Shape{1});
        auto constant = ngraph::opset8::Constant::create(ngraph::element::i8, ngraph::Shape{}, {1});
        auto like = std::make_shared<ngraph::opset8::Add>(data2, constant);
        auto cvtlike = std::make_shared<ngraph::opset8::ConvertLike>(data, like);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{cvtlike}, ngraph::ParameterVector{data, data2});

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<ngraph::pass::ConvertConvertLike>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto data = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{3, 1, 2});
        auto cvt = std::make_shared<ngraph::opset8::Convert>(data, ngraph::element::i8);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{cvt}, ngraph::ParameterVector{data});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertConvertLike_Negative) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto data = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{3, 1, 2});
        auto like = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::dynamic, ngraph::Shape{1});
        auto cvtlike = std::make_shared<ngraph::opset8::ConvertLike>(data, like);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{cvtlike}, ngraph::ParameterVector{data, like});

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<ngraph::pass::ConvertConvertLike>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto data = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{3, 1, 2});
        auto like = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::dynamic, ngraph::Shape{1});
        auto cvtlike = std::make_shared<ngraph::opset8::ConvertLike>(data, like);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{cvtlike}, ngraph::ParameterVector{data, like});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}