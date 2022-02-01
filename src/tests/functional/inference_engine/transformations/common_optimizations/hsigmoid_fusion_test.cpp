// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset7.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/common_optimizations/hsigmoid_fusion.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

TEST_F(TransformationTestsF, HSigmoidFusionWithReluDivF16) {
    {
        auto input = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(1));
        auto add_constant = ngraph::opset7::Constant::create(ngraph::element::f16, ngraph::Shape{}, {3.0});
        auto add = std::make_shared<ngraph::opset7::Add>(input, add_constant);
        auto relu = std::make_shared<ngraph::opset7::Relu>(add);
        auto min_constant = ngraph::opset7::Constant::create(ngraph::element::f16, ngraph::Shape{}, {6.0});
        auto min = std::make_shared<ngraph::opset7::Minimum>(relu, min_constant);
        auto div_constant = ngraph::opset7::Constant::create(ngraph::element::f16, ngraph::Shape{}, {6.0});
        auto div = std::make_shared<ngraph::opset7::Divide>(min, div_constant);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{div}, ngraph::ParameterVector{input});

        manager.register_pass<ngraph::pass::HSigmoidFusion>();
    }

    {
        auto input = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(1));
        auto hsigmoid = std::make_shared<ngraph::opset7::HSigmoid>(input);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{hsigmoid}, ngraph::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, HSigmoidFusionWithReluDivF32) {
    {
        auto input = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32, ngraph::Shape{});
        auto add_constant = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{}, {3.0});
        auto add = std::make_shared<ngraph::opset7::Add>(input, add_constant);
        auto relu = std::make_shared<ngraph::opset7::Relu>(add);
        auto min_constant = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{}, {6.0});
        auto min = std::make_shared<ngraph::opset7::Minimum>(relu, min_constant);
        auto div_constant = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{}, {6.0});
        auto div = std::make_shared<ngraph::opset7::Divide>(min, div_constant);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{div}, ngraph::ParameterVector{input});

        manager.register_pass<ngraph::pass::HSigmoidFusion>();
    }

    {
        auto input = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32, ngraph::Shape{});
        auto hsigmoid = std::make_shared<ngraph::opset7::HSigmoid>(input);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{hsigmoid}, ngraph::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, HSigmoidFusionWithReluMul) {
    {
        auto input = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(1));
        auto add_constant = ngraph::opset7::Constant::create(ngraph::element::f16, ngraph::Shape{}, {3.0});
        auto add = std::make_shared<ngraph::opset7::Add>(input, add_constant);
        auto relu = std::make_shared<ngraph::opset7::Relu>(add);
        auto min_constant = ngraph::opset7::Constant::create(ngraph::element::f16, ngraph::Shape{}, {6.0});
        auto min = std::make_shared<ngraph::opset7::Minimum>(relu, min_constant);
        auto mul_constant = ngraph::opset7::Constant::create(ngraph::element::f16, ngraph::Shape{}, {0.1666666716});
        auto mul_second = std::make_shared<ngraph::opset7::Multiply>(min, mul_constant);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{mul_second}, ngraph::ParameterVector{input});

        manager.register_pass<ngraph::pass::HSigmoidFusion>();
    }

    {
        auto input = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(1));
        auto hsigmoid = std::make_shared<ngraph::opset7::HSigmoid>(input);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{hsigmoid}, ngraph::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, HSigmoidFusionWithoutRelu) {
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

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{div}, ngraph::ParameterVector{input});

        manager.register_pass<ngraph::pass::HSigmoidFusion>();
    }

    {
        auto input = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(1));
        auto hsigmoid = std::make_shared<ngraph::opset7::HSigmoid>(input);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{hsigmoid}, ngraph::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, HSigmoidFusionWithClampMul) {
    {
        auto input = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(1));
        auto add_constant = ngraph::opset7::Constant::create(ngraph::element::f16, ngraph::Shape{}, {3.0});
        auto add = std::make_shared<ngraph::opset7::Add>(input, add_constant);
        auto clamp = std::make_shared<ngraph::op::v0::Clamp>(add, 0.0f, 6.0f);
        auto mul_constant = ngraph::opset7::Constant::create(ngraph::element::f16, ngraph::Shape{}, {1.0 / 6.0});
        auto mul_first = std::make_shared<ngraph::opset7::Multiply>(clamp, mul_constant);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{mul_first}, ngraph::ParameterVector{input});

        manager.register_pass<ngraph::pass::HSigmoidFusion>();
    }

    {
        auto input = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(1));
        auto hsigmoid = std::make_shared<ngraph::opset7::HSigmoid>(input);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{hsigmoid}, ngraph::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, HSigmoidFusionWithClampDiv) {
    {
        auto input = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(1));
        auto add_constant = ngraph::opset6::Constant::create(ngraph::element::f16, ngraph::Shape{}, {3.0});
        auto add = std::make_shared<ngraph::opset6::Add>(input, add_constant);
        auto clamp = std::make_shared<ngraph::opset6::Clamp>(add, 0.0f, 6.0f);
        auto div_constant = ngraph::opset6::Constant::create(ngraph::element::f16, ngraph::Shape{}, {6.0});
        auto div = std::make_shared<ngraph::opset6::Divide>(clamp, div_constant);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{div}, ngraph::ParameterVector{input});

        manager.register_pass<ngraph::pass::HSigmoidFusion>();
    }

    {
        auto input = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(1));
        auto hsigmoid = std::make_shared<ngraph::opset7::HSigmoid>(input);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{hsigmoid}, ngraph::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, HSigmoidFusionWithReluMulWrongConstValue) {
    {
        auto input = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(1));
        auto add_constant = ngraph::opset7::Constant::create(ngraph::element::f16, ngraph::Shape{}, {3.0});
        auto add = std::make_shared<ngraph::opset7::Add>(input, add_constant);
        auto relu = std::make_shared<ngraph::opset7::Relu>(add);
        auto min_constant = ngraph::opset7::Constant::create(ngraph::element::f16, ngraph::Shape{}, {6.0});
        auto min = std::make_shared<ngraph::opset7::Minimum>(relu, min_constant);
        auto mul_constant = ngraph::opset7::Constant::create(ngraph::element::f16, ngraph::Shape{}, {0.167});
        auto mul_second = std::make_shared<ngraph::opset7::Multiply>(min, mul_constant);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{mul_second}, ngraph::ParameterVector{input});

        manager.register_pass<ngraph::pass::HSigmoidFusion>();
    }

    {
        auto input = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(1));
        auto add_constant = ngraph::opset7::Constant::create(ngraph::element::f16, ngraph::Shape{}, {3.0});
        auto add = std::make_shared<ngraph::opset7::Add>(input, add_constant);
        auto relu = std::make_shared<ngraph::opset7::Relu>(add);
        auto min_constant = ngraph::opset7::Constant::create(ngraph::element::f16, ngraph::Shape{}, {6.0});
        auto min = std::make_shared<ngraph::opset7::Minimum>(relu, min_constant);
        auto mul_constant = ngraph::opset7::Constant::create(ngraph::element::f16, ngraph::Shape{}, {0.167});
        auto mul_second = std::make_shared<ngraph::opset7::Multiply>(min, mul_constant);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{mul_second}, ngraph::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, HSigmoidFusionWithReluDivWrongConstValue) {
    {
        auto input = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f16, ngraph::Shape{});
        auto add_constant = ngraph::opset7::Constant::create(ngraph::element::f16, ngraph::Shape{}, {3.01});
        auto add = std::make_shared<ngraph::opset7::Add>(input, add_constant);
        auto relu = std::make_shared<ngraph::opset7::Relu>(add);
        auto min_constant = ngraph::opset7::Constant::create(ngraph::element::f16, ngraph::Shape{}, {6.002});
        auto min = std::make_shared<ngraph::opset7::Minimum>(relu, min_constant);
        auto div_constant = ngraph::opset7::Constant::create(ngraph::element::f16, ngraph::Shape{}, {0.0});
        auto div = std::make_shared<ngraph::opset7::Divide>(min, div_constant);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{div}, ngraph::ParameterVector{input});

        manager.register_pass<ngraph::pass::HSigmoidFusion>();
    }

    {
        auto input = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f16, ngraph::Shape{});
        auto add_constant = ngraph::opset7::Constant::create(ngraph::element::f16, ngraph::Shape{}, {3.01});
        auto add = std::make_shared<ngraph::opset7::Add>(input, add_constant);
        auto relu = std::make_shared<ngraph::opset7::Relu>(add);
        auto min_constant = ngraph::opset7::Constant::create(ngraph::element::f16, ngraph::Shape{}, {6.002});
        auto min = std::make_shared<ngraph::opset7::Minimum>(relu, min_constant);
        auto div_constant = ngraph::opset7::Constant::create(ngraph::element::f16, ngraph::Shape{}, {0.0});
        auto div = std::make_shared<ngraph::opset7::Divide>(min, div_constant);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{div}, ngraph::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, HSigmoidFusionWithoutReluWrongConstValue) {
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

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{div}, ngraph::ParameterVector{input});

        manager.register_pass<ngraph::pass::HSigmoidFusion>();
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

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{div}, ngraph::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, HSigmoidFusionWithClampWrongConstValue) {
    {
        auto input = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(1));
        auto add_constant = ngraph::opset7::Constant::create(ngraph::element::f16, ngraph::Shape{}, {3.11});
        auto add = std::make_shared<ngraph::opset7::Add>(input, add_constant);
        auto clamp = std::make_shared<ngraph::op::v0::Clamp>(add, 0.11f, 6.02f);
        auto mul_constant = ngraph::opset7::Constant::create(ngraph::element::f16, ngraph::Shape{}, {0.98 / 6.15});
        auto mul_first = std::make_shared<ngraph::opset7::Multiply>(clamp, mul_constant);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{mul_first}, ngraph::ParameterVector{input});

        manager.register_pass<ngraph::pass::HSigmoidFusion>();
    }

    {
        auto input = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(1));
        auto add_constant = ngraph::opset7::Constant::create(ngraph::element::f16, ngraph::Shape{}, {3.11});
        auto add = std::make_shared<ngraph::opset7::Add>(input, add_constant);
        auto clamp = std::make_shared<ngraph::op::v0::Clamp>(add, 0.11f, 6.02f);
        auto mul_constant = ngraph::opset7::Constant::create(ngraph::element::f16, ngraph::Shape{}, {0.98 / 6.15});
        auto mul_first = std::make_shared<ngraph::opset7::Multiply>(clamp, mul_constant);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{mul_first}, ngraph::ParameterVector{input});
    }
}
