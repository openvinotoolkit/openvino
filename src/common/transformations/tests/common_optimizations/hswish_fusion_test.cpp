// Copyright (C) 2018-2022 Intel Corporation
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

TEST_F(TransformationTestsF, HSwishFusionWithReluDivF16) {
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

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{div}, ngraph::ParameterVector{input});

        manager.register_pass<ngraph::pass::HSwishFusion>();
    }

    {
        auto input = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(1));
        auto hswish = std::make_shared<ngraph::opset7::HSwish>(input);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{hswish}, ngraph::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, HSwishFusionWithReluDivF32) {
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

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{div}, ngraph::ParameterVector{input});

        manager.register_pass<ngraph::pass::HSwishFusion>();
    }

    {
        auto input = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32, ngraph::Shape{});
        auto hswish = std::make_shared<ngraph::opset7::HSwish>(input);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{hswish}, ngraph::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, HSwishFusionWithReluMul) {
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

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{mul_second}, ngraph::ParameterVector{input});

        manager.register_pass<ngraph::pass::HSwishFusion>();
    }

    {
        auto input = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(1));
        auto hswish = std::make_shared<ngraph::opset7::HSwish>(input);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{hswish}, ngraph::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, HSwishFusionWithoutRelu) {
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

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{mul}, ngraph::ParameterVector{input});

        auto gr = manager.register_pass<ngraph::pass::GraphRewrite>();
        gr->add_matcher<ngraph::pass::HSigmoidFusion>();
        gr->add_matcher<ngraph::pass::HSwishFusion>();
    }

    {
        auto input = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(1));
        auto hswish = std::make_shared<ngraph::opset7::HSwish>(input);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{hswish}, ngraph::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, HSwishFusionWithClampMul) {
    {
        auto input = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(1));
        auto add_constant = ngraph::opset7::Constant::create(ngraph::element::f16, ngraph::Shape{}, {3.0});
        auto add = std::make_shared<ngraph::opset7::Add>(input, add_constant);
        auto clamp = std::make_shared<ngraph::opset7::Clamp>(add, 0.0f, 6.0f);
        auto mul_constant = ngraph::opset7::Constant::create(ngraph::element::f16, ngraph::Shape{}, {1.0 / 6.0});
        auto mul_first = std::make_shared<ngraph::opset7::Multiply>(clamp, mul_constant);
        auto mul_second = std::make_shared<ngraph::opset7::Multiply>(input, mul_first);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{mul_second}, ngraph::ParameterVector{input});

        auto gr = manager.register_pass<ngraph::pass::GraphRewrite>();
        gr->add_matcher<ngraph::pass::HSigmoidFusion>();
        gr->add_matcher<ngraph::pass::HSwishFusion>();
    }

    {
        auto input = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(1));
        auto hswish = std::make_shared<ngraph::opset7::HSwish>(input);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{hswish}, ngraph::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, HSwishFusionWithClampDiv) {
    {
        auto input = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(1));
        auto add_constant = ngraph::opset7::Constant::create(ngraph::element::f16, ngraph::Shape{}, {3.0});
        auto add = std::make_shared<ngraph::opset7::Add>(input, add_constant);
        auto clamp = std::make_shared<ngraph::opset7::Clamp>(add, 0.0f, 6.0f);
        auto div_constant = ngraph::opset7::Constant::create(ngraph::element::f16, ngraph::Shape{}, {6.0});
        auto div = std::make_shared<ngraph::opset7::Divide>(clamp, div_constant);
        auto mul = std::make_shared<ngraph::opset7::Multiply>(input, div);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{mul}, ngraph::ParameterVector{input});

        auto gr = manager.register_pass<ngraph::pass::GraphRewrite>();
        gr->add_matcher<ngraph::pass::HSigmoidFusion>();
        gr->add_matcher<ngraph::pass::HSwishFusion>();
    }

    {
        auto input = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(1));
        auto hswish = std::make_shared<ngraph::opset7::HSwish>(input);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{hswish}, ngraph::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, HSwishFusionWithReluMulWrongConstValue) {
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

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{mul_second}, ngraph::ParameterVector{input});

        manager.register_pass<ngraph::pass::HSwishFusion>();
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

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{mul_second}, ngraph::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, HSwishFusionWithReluDivWrongConstValue) {
    {
        auto input = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f16, ngraph::Shape{});
        auto add_constant = ngraph::opset7::Constant::create(ngraph::element::f16, ngraph::Shape{}, {3.01});
        auto add = std::make_shared<ngraph::opset7::Add>(input, add_constant);
        auto relu = std::make_shared<ngraph::opset7::Relu>(add);
        auto min_constant = ngraph::opset7::Constant::create(ngraph::element::f16, ngraph::Shape{}, {6.002});
        auto min = std::make_shared<ngraph::opset7::Minimum>(relu, min_constant);
        auto mul = std::make_shared<ngraph::opset7::Multiply>(input, min);
        auto div_constant = ngraph::opset7::Constant::create(ngraph::element::f16, ngraph::Shape{}, {0.1});
        auto div = std::make_shared<ngraph::opset7::Divide>(mul, div_constant);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{div}, ngraph::ParameterVector{input});

        manager.register_pass<ngraph::pass::HSwishFusion>();
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

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{div}, ngraph::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, HSwishFusionWithoutReluWrongConstValue) {
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

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{mul}, ngraph::ParameterVector{input});

        auto gr = manager.register_pass<ngraph::pass::GraphRewrite>();
        gr->add_matcher<ngraph::pass::HSigmoidFusion>();
        gr->add_matcher<ngraph::pass::HSwishFusion>();
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

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{mul}, ngraph::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, HSwishFusionWithClampWrongConstValue) {
    {
        auto input = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(1));
        auto add_constant = ngraph::opset7::Constant::create(ngraph::element::f16, ngraph::Shape{}, {3.11});
        auto add = std::make_shared<ngraph::opset7::Add>(input, add_constant);
        auto clamp = std::make_shared<ngraph::opset7::Clamp>(add, 0.11f, 6.02f);
        auto mul_constant = ngraph::opset7::Constant::create(ngraph::element::f16, ngraph::Shape{}, {0.98 / 6.15});
        auto mul_first = std::make_shared<ngraph::opset7::Multiply>(clamp, mul_constant);
        auto mul_second = std::make_shared<ngraph::opset7::Multiply>(input, mul_first);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{mul_second}, ngraph::ParameterVector{input});

        auto gr = manager.register_pass<ngraph::pass::GraphRewrite>();
        gr->add_matcher<ngraph::pass::HSigmoidFusion>();
        gr->add_matcher<ngraph::pass::HSwishFusion>();
    }

    {
        auto input = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(1));
        auto add_constant = ngraph::opset7::Constant::create(ngraph::element::f16, ngraph::Shape{}, {3.11});
        auto add = std::make_shared<ngraph::opset7::Add>(input, add_constant);
        auto clamp = std::make_shared<ngraph::opset7::Clamp>(add, 0.11f, 6.02f);
        auto mul_constant = ngraph::opset7::Constant::create(ngraph::element::f16, ngraph::Shape{}, {0.98 / 6.15});
        auto mul_first = std::make_shared<ngraph::opset7::Multiply>(clamp, mul_constant);
        auto mul_second = std::make_shared<ngraph::opset7::Multiply>(input, mul_first);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{mul_second}, ngraph::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, HSwishFusionWithHSigmoidMul) {
    {
        auto input = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(1));
        auto hsigmoid = std::make_shared<ngraph::opset7::HSigmoid>(input);
        auto mul = std::make_shared<ngraph::opset7::Multiply>(input, hsigmoid);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{mul}, ngraph::ParameterVector{input});

        manager.register_pass<ngraph::pass::HSwishFusion>();
    }

    {
        auto input = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(1));
        auto hswish = std::make_shared<ngraph::opset7::HSwish>(input);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{hswish}, ngraph::ParameterVector{input});
    }
}


TEST_F(TransformationTestsF, HSwishFusionWithClamp) {
    {
        auto input = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(1));
        auto add_constant = ngraph::opset7::Constant::create(ngraph::element::f16, ngraph::Shape{}, {3.0});
        auto add = std::make_shared<ngraph::opset7::Add>(input, add_constant);
        auto clamp = std::make_shared<ngraph::opset7::Clamp>(add, 0.0f, 6.0f);
        auto mul = std::make_shared<ngraph::opset7::Multiply>(input, clamp);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{mul}, ngraph::ParameterVector{input});

        auto gr = manager.register_pass<ngraph::pass::GraphRewrite>();
        gr->add_matcher<ngraph::pass::HSwishFusion>();
    }

    {
        auto input = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(1));
        auto hswish = std::make_shared<ngraph::opset7::HSwish>(input);
        auto mul_const = ngraph::opset7::Constant::create(ngraph::element::f16, ngraph::Shape{}, {6.0});
        auto mul = std::make_shared<ngraph::opset7::Multiply>(hswish, mul_const);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{mul}, ngraph::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, HSwishFusionWithClampWithWrongConstant) {
    {
        auto input = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(1));
        auto add_constant = ngraph::opset7::Constant::create(ngraph::element::f16, ngraph::Shape{}, {3.11});
        auto add = std::make_shared<ngraph::opset7::Add>(input, add_constant);
        auto clamp = std::make_shared<ngraph::opset7::Clamp>(add, 0.11f, 6.32f);
        auto mul = std::make_shared<ngraph::opset7::Multiply>(input, clamp);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{mul}, ngraph::ParameterVector{input});

        auto gr = manager.register_pass<ngraph::pass::GraphRewrite>();
        gr->add_matcher<ngraph::pass::HSwishFusion>();
    }

    {
        auto input = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(1));
        auto add_constant = ngraph::opset7::Constant::create(ngraph::element::f16, ngraph::Shape{}, {3.11});
        auto add = std::make_shared<ngraph::opset7::Add>(input, add_constant);
        auto clamp = std::make_shared<ngraph::opset7::Clamp>(add, 0.11f, 6.32f);
        auto mul = std::make_shared<ngraph::opset7::Multiply>(input, clamp);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{mul}, ngraph::ParameterVector{input});
    }
}