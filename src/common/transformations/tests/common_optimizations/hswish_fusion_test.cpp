// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/hswish_fusion.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset7.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/common_optimizations/hsigmoid_fusion.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov;
using namespace testing;

TEST_F(TransformationTestsF, HSwishFusionWithReluDivF16) {
    {
        auto input = std::make_shared<opset7::Parameter>(element::f16, PartialShape::dynamic(1));
        auto add_constant = opset7::Constant::create(element::f16, Shape{}, {3.0});
        auto add = std::make_shared<opset7::Add>(input, add_constant);
        auto relu = std::make_shared<opset7::Relu>(add);
        auto min_constant = opset7::Constant::create(element::f16, Shape{}, {6.0});
        auto min = std::make_shared<opset7::Minimum>(relu, min_constant);
        auto mul = std::make_shared<opset7::Multiply>(input, min);
        auto div_constant = opset7::Constant::create(element::f16, Shape{}, {6.0});
        auto div = std::make_shared<opset7::Divide>(mul, div_constant);

        model = std::make_shared<ov::Model>(NodeVector{div}, ParameterVector{input});

        manager.register_pass<ov::pass::HSwishFusion>();
    }

    {
        auto input = std::make_shared<opset7::Parameter>(element::f16, PartialShape::dynamic(1));
        auto hswish = std::make_shared<opset7::HSwish>(input);

        model_ref = std::make_shared<ov::Model>(NodeVector{hswish}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, HSwishFusionWithReluDivF32) {
    {
        auto input = std::make_shared<opset7::Parameter>(element::f32, Shape{});
        auto add_constant = opset7::Constant::create(element::f32, Shape{}, {3.0});
        auto add = std::make_shared<opset7::Add>(input, add_constant);
        auto relu = std::make_shared<opset7::Relu>(add);
        auto min_constant = opset7::Constant::create(element::f32, Shape{}, {6.0});
        auto min = std::make_shared<opset7::Minimum>(relu, min_constant);
        auto mul = std::make_shared<opset7::Multiply>(input, min);
        auto div_constant = opset7::Constant::create(element::f32, Shape{}, {6.0});
        auto div = std::make_shared<opset7::Divide>(mul, div_constant);

        model = std::make_shared<ov::Model>(NodeVector{div}, ParameterVector{input});

        manager.register_pass<ov::pass::HSwishFusion>();
    }

    {
        auto input = std::make_shared<opset7::Parameter>(element::f32, Shape{});
        auto hswish = std::make_shared<opset7::HSwish>(input);

        model_ref = std::make_shared<ov::Model>(NodeVector{hswish}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, HSwishFusionWithReluMul) {
    {
        auto input = std::make_shared<opset7::Parameter>(element::f16, PartialShape::dynamic(1));
        auto add_constant = opset7::Constant::create(element::f16, Shape{}, {3.0});
        auto add = std::make_shared<opset7::Add>(input, add_constant);
        auto relu = std::make_shared<opset7::Relu>(add);
        auto min_constant = opset7::Constant::create(element::f16, Shape{}, {6.0});
        auto min = std::make_shared<opset7::Minimum>(relu, min_constant);
        auto mul_first = std::make_shared<opset7::Multiply>(input, min);
        auto mul_constant = opset7::Constant::create(element::f16, Shape{}, {0.1666666716});
        auto mul_second = std::make_shared<opset7::Multiply>(mul_first, mul_constant);

        model = std::make_shared<ov::Model>(NodeVector{mul_second}, ParameterVector{input});

        manager.register_pass<ov::pass::HSwishFusion>();
    }

    {
        auto input = std::make_shared<opset7::Parameter>(element::f16, PartialShape::dynamic(1));
        auto hswish = std::make_shared<opset7::HSwish>(input);

        model_ref = std::make_shared<ov::Model>(NodeVector{hswish}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, HSwishFusionWithoutRelu) {
    {
        auto input = std::make_shared<opset7::Parameter>(element::f16, PartialShape::dynamic(1));
        auto add_constant = opset7::Constant::create(element::f16, Shape{}, {3.0});
        auto add = std::make_shared<opset7::Add>(input, add_constant);
        auto max_constant = opset7::Constant::create(element::f16, Shape{}, {0.0});
        auto max = std::make_shared<opset7::Maximum>(add, max_constant);
        auto min_constant = opset7::Constant::create(element::f16, Shape{}, {6.0});
        auto min = std::make_shared<opset7::Minimum>(max, min_constant);
        auto div_constant = opset7::Constant::create(element::f16, Shape{}, {6.0});
        auto div = std::make_shared<opset7::Divide>(min, div_constant);
        auto mul = std::make_shared<opset7::Multiply>(input, div);

        model = std::make_shared<ov::Model>(NodeVector{mul}, ParameterVector{input});

        auto gr = manager.register_pass<pass::GraphRewrite>();
        gr->add_matcher<ov::pass::HSigmoidFusion>();
        gr->add_matcher<ov::pass::HSwishFusion>();
    }

    {
        auto input = std::make_shared<opset7::Parameter>(element::f16, PartialShape::dynamic(1));
        auto hswish = std::make_shared<opset7::HSwish>(input);

        model_ref = std::make_shared<ov::Model>(NodeVector{hswish}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, HSwishFusionWithClampMul) {
    {
        auto input = std::make_shared<opset7::Parameter>(element::f16, PartialShape::dynamic(1));
        auto add_constant = opset7::Constant::create(element::f16, Shape{}, {3.0});
        auto add = std::make_shared<opset7::Add>(input, add_constant);
        auto clamp = std::make_shared<opset7::Clamp>(add, 0.0f, 6.0f);
        auto mul_constant = opset7::Constant::create(element::f16, Shape{}, {1.0 / 6.0});
        auto mul_first = std::make_shared<opset7::Multiply>(clamp, mul_constant);
        auto mul_second = std::make_shared<opset7::Multiply>(input, mul_first);

        model = std::make_shared<ov::Model>(NodeVector{mul_second}, ParameterVector{input});

        auto gr = manager.register_pass<pass::GraphRewrite>();
        gr->add_matcher<ov::pass::HSigmoidFusion>();
        gr->add_matcher<ov::pass::HSwishFusion>();
    }

    {
        auto input = std::make_shared<opset7::Parameter>(element::f16, PartialShape::dynamic(1));
        auto hswish = std::make_shared<opset7::HSwish>(input);

        model_ref = std::make_shared<ov::Model>(NodeVector{hswish}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, HSwishFusionWithClampDiv) {
    {
        auto input = std::make_shared<opset7::Parameter>(element::f16, PartialShape::dynamic(1));
        auto add_constant = opset7::Constant::create(element::f16, Shape{}, {3.0});
        auto add = std::make_shared<opset7::Add>(input, add_constant);
        auto clamp = std::make_shared<opset7::Clamp>(add, 0.0f, 6.0f);
        auto div_constant = opset7::Constant::create(element::f16, Shape{}, {6.0});
        auto div = std::make_shared<opset7::Divide>(clamp, div_constant);
        auto mul = std::make_shared<opset7::Multiply>(input, div);

        model = std::make_shared<ov::Model>(NodeVector{mul}, ParameterVector{input});

        auto gr = manager.register_pass<pass::GraphRewrite>();
        gr->add_matcher<ov::pass::HSigmoidFusion>();
        gr->add_matcher<ov::pass::HSwishFusion>();
    }

    {
        auto input = std::make_shared<opset7::Parameter>(element::f16, PartialShape::dynamic(1));
        auto hswish = std::make_shared<opset7::HSwish>(input);

        model_ref = std::make_shared<ov::Model>(NodeVector{hswish}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, HSwishFusionWithReluMulWrongConstValue) {
    {
        auto input = std::make_shared<opset7::Parameter>(element::f16, PartialShape::dynamic(1));
        auto add_constant = opset7::Constant::create(element::f16, Shape{}, {3.0});
        auto add = std::make_shared<opset7::Add>(input, add_constant);
        auto relu = std::make_shared<opset7::Relu>(add);
        auto min_constant = opset7::Constant::create(element::f16, Shape{}, {6.0});
        auto min = std::make_shared<opset7::Minimum>(relu, min_constant);
        auto mul_first = std::make_shared<opset7::Multiply>(input, min);
        auto mul_constant = opset7::Constant::create(element::f16, Shape{}, {0.167});
        auto mul_second = std::make_shared<opset7::Multiply>(mul_first, mul_constant);

        model = std::make_shared<ov::Model>(NodeVector{mul_second}, ParameterVector{input});

        manager.register_pass<ov::pass::HSwishFusion>();
    }

    {
        auto input = std::make_shared<opset7::Parameter>(element::f16, PartialShape::dynamic(1));
        auto add_constant = opset7::Constant::create(element::f16, Shape{}, {3.0});
        auto add = std::make_shared<opset7::Add>(input, add_constant);
        auto relu = std::make_shared<opset7::Relu>(add);
        auto min_constant = opset7::Constant::create(element::f16, Shape{}, {6.0});
        auto min = std::make_shared<opset7::Minimum>(relu, min_constant);
        auto mul_first = std::make_shared<opset7::Multiply>(input, min);
        auto mul_constant = opset7::Constant::create(element::f16, Shape{}, {0.167});
        auto mul_second = std::make_shared<opset7::Multiply>(mul_first, mul_constant);

        model_ref = std::make_shared<ov::Model>(NodeVector{mul_second}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, HSwishFusionWithReluDivWrongConstValue) {
    {
        auto input = std::make_shared<opset7::Parameter>(element::f16, Shape{});
        auto add_constant = opset7::Constant::create(element::f16, Shape{}, {3.01});
        auto add = std::make_shared<opset7::Add>(input, add_constant);
        auto relu = std::make_shared<opset7::Relu>(add);
        auto min_constant = opset7::Constant::create(element::f16, Shape{}, {6.002});
        auto min = std::make_shared<opset7::Minimum>(relu, min_constant);
        auto mul = std::make_shared<opset7::Multiply>(input, min);
        auto div_constant = opset7::Constant::create(element::f16, Shape{}, {0.1});
        auto div = std::make_shared<opset7::Divide>(mul, div_constant);

        model = std::make_shared<ov::Model>(NodeVector{div}, ParameterVector{input});

        manager.register_pass<ov::pass::HSwishFusion>();
    }

    {
        auto input = std::make_shared<opset7::Parameter>(element::f16, Shape{});
        auto add_constant = opset7::Constant::create(element::f16, Shape{}, {3.01});
        auto add = std::make_shared<opset7::Add>(input, add_constant);
        auto relu = std::make_shared<opset7::Relu>(add);
        auto min_constant = opset7::Constant::create(element::f16, Shape{}, {6.002});
        auto min = std::make_shared<opset7::Minimum>(relu, min_constant);
        auto mul = std::make_shared<opset7::Multiply>(input, min);
        auto div_constant = opset7::Constant::create(element::f16, Shape{}, {0.0});
        auto div = std::make_shared<opset7::Divide>(mul, div_constant);

        model_ref = std::make_shared<ov::Model>(NodeVector{div}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, HSwishFusionWithoutReluWrongConstValue) {
    {
        auto input = std::make_shared<opset7::Parameter>(element::f16, PartialShape::dynamic(1));
        auto add_constant = opset7::Constant::create(element::f16, Shape{}, {3.11});
        auto add = std::make_shared<opset7::Add>(input, add_constant);
        auto max_constant = opset7::Constant::create(element::f16, Shape{}, {0.22});
        auto max = std::make_shared<opset7::Maximum>(add, max_constant);
        auto min_constant = opset7::Constant::create(element::f16, Shape{}, {6.01});
        auto min = std::make_shared<opset7::Minimum>(max, min_constant);
        auto div_constant = opset7::Constant::create(element::f16, Shape{}, {6.002});
        auto div = std::make_shared<opset7::Divide>(min, div_constant);
        auto mul = std::make_shared<opset7::Multiply>(input, div);

        model = std::make_shared<ov::Model>(NodeVector{mul}, ParameterVector{input});

        auto gr = manager.register_pass<pass::GraphRewrite>();
        gr->add_matcher<ov::pass::HSigmoidFusion>();
        gr->add_matcher<ov::pass::HSwishFusion>();
    }

    {
        auto input = std::make_shared<opset7::Parameter>(element::f16, PartialShape::dynamic(1));
        auto add_constant = opset7::Constant::create(element::f16, Shape{}, {3.11});
        auto add = std::make_shared<opset7::Add>(input, add_constant);
        auto max_constant = opset7::Constant::create(element::f16, Shape{}, {0.22});
        auto max = std::make_shared<opset7::Maximum>(add, max_constant);
        auto min_constant = opset7::Constant::create(element::f16, Shape{}, {6.01});
        auto min = std::make_shared<opset7::Minimum>(max, min_constant);
        auto div_constant = opset7::Constant::create(element::f16, Shape{}, {6.002});
        auto div = std::make_shared<opset7::Divide>(min, div_constant);
        auto mul = std::make_shared<opset7::Multiply>(input, div);

        model_ref = std::make_shared<ov::Model>(NodeVector{mul}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, HSwishFusionWithClampWrongConstValue) {
    {
        auto input = std::make_shared<opset7::Parameter>(element::f16, PartialShape::dynamic(1));
        auto add_constant = opset7::Constant::create(element::f16, Shape{}, {3.11});
        auto add = std::make_shared<opset7::Add>(input, add_constant);
        auto clamp = std::make_shared<opset7::Clamp>(add, 0.11f, 6.02f);
        auto mul_constant = opset7::Constant::create(element::f16, Shape{}, {0.98 / 6.15});
        auto mul_first = std::make_shared<opset7::Multiply>(clamp, mul_constant);
        auto mul_second = std::make_shared<opset7::Multiply>(input, mul_first);

        model = std::make_shared<ov::Model>(NodeVector{mul_second}, ParameterVector{input});

        auto gr = manager.register_pass<pass::GraphRewrite>();
        gr->add_matcher<ov::pass::HSigmoidFusion>();
        gr->add_matcher<ov::pass::HSwishFusion>();
    }

    {
        auto input = std::make_shared<opset7::Parameter>(element::f16, PartialShape::dynamic(1));
        auto add_constant = opset7::Constant::create(element::f16, Shape{}, {3.11});
        auto add = std::make_shared<opset7::Add>(input, add_constant);
        auto clamp = std::make_shared<opset7::Clamp>(add, 0.11f, 6.02f);
        auto mul_constant = opset7::Constant::create(element::f16, Shape{}, {0.98 / 6.15});
        auto mul_first = std::make_shared<opset7::Multiply>(clamp, mul_constant);
        auto mul_second = std::make_shared<opset7::Multiply>(input, mul_first);

        model_ref = std::make_shared<ov::Model>(NodeVector{mul_second}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, HSwishFusionWithHSigmoidMul) {
    {
        auto input = std::make_shared<opset7::Parameter>(element::f16, PartialShape::dynamic(1));
        auto hsigmoid = std::make_shared<opset7::HSigmoid>(input);
        auto mul = std::make_shared<opset7::Multiply>(input, hsigmoid);

        model = std::make_shared<ov::Model>(NodeVector{mul}, ParameterVector{input});

        manager.register_pass<ov::pass::HSwishFusion>();
    }

    {
        auto input = std::make_shared<opset7::Parameter>(element::f16, PartialShape::dynamic(1));
        auto hswish = std::make_shared<opset7::HSwish>(input);

        model_ref = std::make_shared<ov::Model>(NodeVector{hswish}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, HSwishFusionWithClamp) {
    {
        auto input = std::make_shared<opset7::Parameter>(element::f16, PartialShape::dynamic(1));
        auto add_constant = opset7::Constant::create(element::f16, Shape{}, {3.0});
        auto add = std::make_shared<opset7::Add>(input, add_constant);
        auto clamp = std::make_shared<opset7::Clamp>(add, 0.0f, 6.0f);
        auto mul = std::make_shared<opset7::Multiply>(input, clamp);

        model = std::make_shared<ov::Model>(NodeVector{mul}, ParameterVector{input});

        auto gr = manager.register_pass<pass::GraphRewrite>();
        gr->add_matcher<ov::pass::HSwishFusion>();
    }

    {
        auto input = std::make_shared<opset7::Parameter>(element::f16, PartialShape::dynamic(1));
        auto hswish = std::make_shared<opset7::HSwish>(input);
        auto mul_const = opset7::Constant::create(element::f16, Shape{}, {6.0});
        auto mul = std::make_shared<opset7::Multiply>(hswish, mul_const);

        model_ref = std::make_shared<ov::Model>(NodeVector{mul}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, HSwishFusionWithClampWithWrongConstant) {
    {
        auto input = std::make_shared<opset7::Parameter>(element::f16, PartialShape::dynamic(1));
        auto add_constant = opset7::Constant::create(element::f16, Shape{}, {3.11});
        auto add = std::make_shared<opset7::Add>(input, add_constant);
        auto clamp = std::make_shared<opset7::Clamp>(add, 0.11f, 6.32f);
        auto mul = std::make_shared<opset7::Multiply>(input, clamp);

        model = std::make_shared<ov::Model>(NodeVector{mul}, ParameterVector{input});

        auto gr = manager.register_pass<pass::GraphRewrite>();
        gr->add_matcher<ov::pass::HSwishFusion>();
    }

    {
        auto input = std::make_shared<opset7::Parameter>(element::f16, PartialShape::dynamic(1));
        auto add_constant = opset7::Constant::create(element::f16, Shape{}, {3.11});
        auto add = std::make_shared<opset7::Add>(input, add_constant);
        auto clamp = std::make_shared<opset7::Clamp>(add, 0.11f, 6.32f);
        auto mul = std::make_shared<opset7::Multiply>(input, clamp);

        model_ref = std::make_shared<ov::Model>(NodeVector{mul}, ParameterVector{input});
    }
}
