// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset7.hpp>
#include <ngraph/pass/manager.hpp>
#include "transformations/common_optimizations/hsigmoid_fusion.hpp"
#include <transformations/common_optimizations/hswish_fusion.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

TEST(TransformationTests, HSwishFusionWithReluDivF16) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(1));
        auto add_constant = ngraph::opset7::Constant::create(ngraph::element::f16, ngraph::Shape{}, {3.0});
        auto add = std::make_shared<ngraph::opset7::Add>(input, add_constant);
        auto relu = std::make_shared<ngraph::opset7::Relu>(add);
        auto min_constant = ngraph::opset7::Constant::create(ngraph::element::f16, ngraph::Shape{}, {6.0});
        auto min = std::make_shared<ngraph::opset7::Minimum>(relu, min_constant);
        auto mul = std::make_shared<ngraph::opset7::Multiply>(input, min);
        auto div_constant = ngraph::opset7::Constant::create(ngraph::element::f16, ngraph::Shape{}, {6.0});
        auto div = std::make_shared<ngraph::opset7::Divide>(mul, div_constant);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{div}, ngraph::ParameterVector{input});

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::HSwishFusion>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(1));
        auto hswish = std::make_shared<ngraph::opset7::HSwish>(input);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{hswish}, ngraph::ParameterVector{input});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, HSwishFusionWithReluDivF32) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32, ngraph::Shape{});
        auto add_constant = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{}, {3.0});
        auto add = std::make_shared<ngraph::opset7::Add>(input, add_constant);
        auto relu = std::make_shared<ngraph::opset7::Relu>(add);
        auto min_constant = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{}, {6.0});
        auto min = std::make_shared<ngraph::opset7::Minimum>(relu, min_constant);
        auto mul = std::make_shared<ngraph::opset7::Multiply>(input, min);
        auto div_constant = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{}, {6.0});
        auto div = std::make_shared<ngraph::opset7::Divide>(mul, div_constant);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{div}, ngraph::ParameterVector{input});

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::HSwishFusion>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32, ngraph::Shape{});
        auto hswish = std::make_shared<ngraph::opset7::HSwish>(input);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{hswish}, ngraph::ParameterVector{input});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, HSwishFusionWithReluMul) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(1));
        auto add_constant = ngraph::opset7::Constant::create(ngraph::element::f16, ngraph::Shape{}, {3.0});
        auto add = std::make_shared<ngraph::opset7::Add>(input, add_constant);
        auto relu = std::make_shared<ngraph::opset7::Relu>(add);
        auto min_constant = ngraph::opset7::Constant::create(ngraph::element::f16, ngraph::Shape{}, {6.0});
        auto min = std::make_shared<ngraph::opset7::Minimum>(relu, min_constant);
        auto mul_first = std::make_shared<ngraph::opset7::Multiply>(input, min);
        auto mul_constant = ngraph::opset7::Constant::create(ngraph::element::f16, ngraph::Shape{}, {0.1666666716});
        auto mul_second = std::make_shared<ngraph::opset7::Multiply>(mul_first, mul_constant);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{mul_second}, ngraph::ParameterVector{input});

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::HSwishFusion>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(1));
        auto hswish = std::make_shared<ngraph::opset7::HSwish>(input);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{hswish}, ngraph::ParameterVector{input});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, HSwishFusionWithoutRelu) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(1));
        auto add_constant = ngraph::opset7::Constant::create(ngraph::element::f16, ngraph::Shape{}, {3.0});
        auto add = std::make_shared<ngraph::opset7::Add>(input, add_constant);
        auto max_constant = ngraph::opset7::Constant::create(ngraph::element::f16, ngraph::Shape{}, {0.0});
        auto max = std::make_shared<ngraph::opset7::Maximum>(add, max_constant);
        auto min_constant = ngraph::opset7::Constant::create(ngraph::element::f16, ngraph::Shape{}, {6.0});
        auto min = std::make_shared<ngraph::opset7::Minimum>(max, min_constant);
        auto div_constant = ngraph::opset7::Constant::create(ngraph::element::f16, ngraph::Shape{}, {6.0});
        auto div = std::make_shared<ngraph::opset7::Divide>(min, div_constant);
        auto mul = std::make_shared<ngraph::opset7::Multiply>(input, div);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{mul}, ngraph::ParameterVector{input});

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        auto gr = manager.register_pass<ngraph::pass::GraphRewrite>();
        gr->add_matcher<ngraph::pass::HSigmoidFusion>();
        gr->add_matcher<ngraph::pass::HSwishFusion>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(1));
        auto hswish = std::make_shared<ngraph::opset7::HSwish>(input);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{hswish}, ngraph::ParameterVector{input});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, HSwishFusionWithClampMul) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(1));
        auto add_constant = ngraph::opset7::Constant::create(ngraph::element::f16, ngraph::Shape{}, {3.0});
        auto add = std::make_shared<ngraph::opset7::Add>(input, add_constant);
        auto clamp = std::make_shared<ngraph::opset7::Clamp>(add, 0.0f, 6.0f);
        auto mul_constant = ngraph::opset7::Constant::create(ngraph::element::f16, ngraph::Shape{}, {1.0 / 6.0});
        auto mul_first = std::make_shared<ngraph::opset7::Multiply>(clamp, mul_constant);
        auto mul_second = std::make_shared<ngraph::opset7::Multiply>(input, mul_first);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{mul_second}, ngraph::ParameterVector{input});

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        auto gr = manager.register_pass<ngraph::pass::GraphRewrite>();
        gr->add_matcher<ngraph::pass::HSigmoidFusion>();
        gr->add_matcher<ngraph::pass::HSwishFusion>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(1));
        auto hswish = std::make_shared<ngraph::opset7::HSwish>(input);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{hswish}, ngraph::ParameterVector{input});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, HSwishFusionWithClampDiv) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(1));
        auto add_constant = ngraph::opset7::Constant::create(ngraph::element::f16, ngraph::Shape{}, {3.0});
        auto add = std::make_shared<ngraph::opset7::Add>(input, add_constant);
        auto clamp = std::make_shared<ngraph::opset7::Clamp>(add, 0.0f, 6.0f);
        auto div_constant = ngraph::opset7::Constant::create(ngraph::element::f16, ngraph::Shape{}, {6.0});
        auto div = std::make_shared<ngraph::opset7::Divide>(clamp, div_constant);
        auto mul = std::make_shared<ngraph::opset7::Multiply>(input, div);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{mul}, ngraph::ParameterVector{input});

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        auto gr = manager.register_pass<ngraph::pass::GraphRewrite>();
        gr->add_matcher<ngraph::pass::HSigmoidFusion>();
        gr->add_matcher<ngraph::pass::HSwishFusion>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(1));
        auto hswish = std::make_shared<ngraph::opset7::HSwish>(input);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{hswish}, ngraph::ParameterVector{input});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, HSwishFusionWithReluMulWrongConstValue) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(1));
        auto add_constant = ngraph::opset7::Constant::create(ngraph::element::f16, ngraph::Shape{}, {3.0});
        auto add = std::make_shared<ngraph::opset7::Add>(input, add_constant);
        auto relu = std::make_shared<ngraph::opset7::Relu>(add);
        auto min_constant = ngraph::opset7::Constant::create(ngraph::element::f16, ngraph::Shape{}, {6.0});
        auto min = std::make_shared<ngraph::opset7::Minimum>(relu, min_constant);
        auto mul_first = std::make_shared<ngraph::opset7::Multiply>(input, min);
        auto mul_constant = ngraph::opset7::Constant::create(ngraph::element::f16, ngraph::Shape{}, {0.167});
        auto mul_second = std::make_shared<ngraph::opset7::Multiply>(mul_first, mul_constant);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{mul_second}, ngraph::ParameterVector{input});

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::HSwishFusion>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(1));
        auto add_constant = ngraph::opset7::Constant::create(ngraph::element::f16, ngraph::Shape{}, {3.0});
        auto add = std::make_shared<ngraph::opset7::Add>(input, add_constant);
        auto relu = std::make_shared<ngraph::opset7::Relu>(add);
        auto min_constant = ngraph::opset7::Constant::create(ngraph::element::f16, ngraph::Shape{}, {6.0});
        auto min = std::make_shared<ngraph::opset7::Minimum>(relu, min_constant);
        auto mul_first = std::make_shared<ngraph::opset7::Multiply>(input, min);
        auto mul_constant = ngraph::opset7::Constant::create(ngraph::element::f16, ngraph::Shape{}, {0.167});
        auto mul_second = std::make_shared<ngraph::opset7::Multiply>(mul_first, mul_constant);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{mul_second}, ngraph::ParameterVector{input});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, HSwishFusionWithReluDivWrongConstValue) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f16, ngraph::Shape{});
        auto add_constant = ngraph::opset7::Constant::create(ngraph::element::f16, ngraph::Shape{}, {3.01});
        auto add = std::make_shared<ngraph::opset7::Add>(input, add_constant);
        auto relu = std::make_shared<ngraph::opset7::Relu>(add);
        auto min_constant = ngraph::opset7::Constant::create(ngraph::element::f16, ngraph::Shape{}, {6.002});
        auto min = std::make_shared<ngraph::opset7::Minimum>(relu, min_constant);
        auto mul = std::make_shared<ngraph::opset7::Multiply>(input, min);
        auto div_constant = ngraph::opset7::Constant::create(ngraph::element::f16, ngraph::Shape{}, {0.0});
        auto div = std::make_shared<ngraph::opset7::Divide>(mul, div_constant);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{div}, ngraph::ParameterVector{input});

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::HSwishFusion>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f16, ngraph::Shape{});
        auto add_constant = ngraph::opset7::Constant::create(ngraph::element::f16, ngraph::Shape{}, {3.01});
        auto add = std::make_shared<ngraph::opset7::Add>(input, add_constant);
        auto relu = std::make_shared<ngraph::opset7::Relu>(add);
        auto min_constant = ngraph::opset7::Constant::create(ngraph::element::f16, ngraph::Shape{}, {6.002});
        auto min = std::make_shared<ngraph::opset7::Minimum>(relu, min_constant);
        auto mul = std::make_shared<ngraph::opset7::Multiply>(input, min);
        auto div_constant = ngraph::opset7::Constant::create(ngraph::element::f16, ngraph::Shape{}, {0.0});
        auto div = std::make_shared<ngraph::opset7::Divide>(mul, div_constant);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{div}, ngraph::ParameterVector{input});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, HSwishFusionWithoutReluWrongConstValue) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(1));
        auto add_constant = ngraph::opset7::Constant::create(ngraph::element::f16, ngraph::Shape{}, {3.11});
        auto add = std::make_shared<ngraph::opset7::Add>(input, add_constant);
        auto max_constant = ngraph::opset7::Constant::create(ngraph::element::f16, ngraph::Shape{}, {0.22});
        auto max = std::make_shared<ngraph::opset7::Maximum>(add, max_constant);
        auto min_constant = ngraph::opset7::Constant::create(ngraph::element::f16, ngraph::Shape{}, {6.01});
        auto min = std::make_shared<ngraph::opset7::Minimum>(max, min_constant);
        auto div_constant = ngraph::opset7::Constant::create(ngraph::element::f16, ngraph::Shape{}, {6.002});
        auto div = std::make_shared<ngraph::opset7::Divide>(min, div_constant);
        auto mul = std::make_shared<ngraph::opset7::Multiply>(input, div);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{mul}, ngraph::ParameterVector{input});

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        auto gr = manager.register_pass<ngraph::pass::GraphRewrite>();
        gr->add_matcher<ngraph::pass::HSigmoidFusion>();
        gr->add_matcher<ngraph::pass::HSwishFusion>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(1));
        auto add_constant = ngraph::opset7::Constant::create(ngraph::element::f16, ngraph::Shape{}, {3.11});
        auto add = std::make_shared<ngraph::opset7::Add>(input, add_constant);
        auto max_constant = ngraph::opset7::Constant::create(ngraph::element::f16, ngraph::Shape{}, {0.22});
        auto max = std::make_shared<ngraph::opset7::Maximum>(add, max_constant);
        auto min_constant = ngraph::opset7::Constant::create(ngraph::element::f16, ngraph::Shape{}, {6.01});
        auto min = std::make_shared<ngraph::opset7::Minimum>(max, min_constant);
        auto div_constant = ngraph::opset7::Constant::create(ngraph::element::f16, ngraph::Shape{}, {6.002});
        auto div = std::make_shared<ngraph::opset7::Divide>(min, div_constant);
        auto mul = std::make_shared<ngraph::opset7::Multiply>(input, div);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{mul}, ngraph::ParameterVector{input});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, HSwishFusionWithClampWrongConstValue) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(1));
        auto add_constant = ngraph::opset7::Constant::create(ngraph::element::f16, ngraph::Shape{}, {3.11});
        auto add = std::make_shared<ngraph::opset7::Add>(input, add_constant);
        auto clamp = std::make_shared<ngraph::opset7::Clamp>(add, 0.11f, 6.02f);
        auto mul_constant = ngraph::opset7::Constant::create(ngraph::element::f16, ngraph::Shape{}, {0.98 / 6.15});
        auto mul_first = std::make_shared<ngraph::opset7::Multiply>(clamp, mul_constant);
        auto mul_second = std::make_shared<ngraph::opset7::Multiply>(input, mul_first);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{mul_second}, ngraph::ParameterVector{input});

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        auto gr = manager.register_pass<ngraph::pass::GraphRewrite>();
        gr->add_matcher<ngraph::pass::HSigmoidFusion>();
        gr->add_matcher<ngraph::pass::HSwishFusion>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(1));
        auto add_constant = ngraph::opset7::Constant::create(ngraph::element::f16, ngraph::Shape{}, {3.11});
        auto add = std::make_shared<ngraph::opset7::Add>(input, add_constant);
        auto clamp = std::make_shared<ngraph::opset7::Clamp>(add, 0.11f, 6.02f);
        auto mul_constant = ngraph::opset7::Constant::create(ngraph::element::f16, ngraph::Shape{}, {0.98 / 6.15});
        auto mul_first = std::make_shared<ngraph::opset7::Multiply>(clamp, mul_constant);
        auto mul_second = std::make_shared<ngraph::opset7::Multiply>(input, mul_first);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{mul_second}, ngraph::ParameterVector{input});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, HSwishFusionWithHSigmoidMul) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(1));
        auto hsigmoid = std::make_shared<ngraph::opset7::HSigmoid>(input);
        auto mul = std::make_shared<ngraph::opset7::Multiply>(input, hsigmoid);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{mul}, ngraph::ParameterVector{input});

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::HSwishFusion>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(1));
        auto hswish = std::make_shared<ngraph::opset7::HSwish>(input);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{hswish}, ngraph::ParameterVector{input});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}
