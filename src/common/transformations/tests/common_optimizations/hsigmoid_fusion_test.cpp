// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/hsigmoid_fusion.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset7.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov;
using namespace testing;

TEST_F(TransformationTestsF, HSigmoidFusionWithReluDivF16) {
    {
        auto input = std::make_shared<opset7::Parameter>(element::f16, PartialShape::dynamic(1));
        auto add_constant = opset7::Constant::create(element::f16, Shape{}, {3.0});
        auto add = std::make_shared<opset7::Add>(input, add_constant);
        auto relu = std::make_shared<opset7::Relu>(add);
        auto min_constant = opset7::Constant::create(element::f16, Shape{}, {6.0});
        auto min = std::make_shared<opset7::Minimum>(relu, min_constant);
        auto div_constant = opset7::Constant::create(element::f16, Shape{}, {6.0});
        auto div = std::make_shared<opset7::Divide>(min, div_constant);

        model = std::make_shared<ov::Model>(NodeVector{div}, ParameterVector{input});

        manager.register_pass<ov::pass::HSigmoidFusion>();
    }

    {
        auto input = std::make_shared<opset7::Parameter>(element::f16, PartialShape::dynamic(1));
        auto hsigmoid = std::make_shared<opset7::HSigmoid>(input);

        model_ref = std::make_shared<ov::Model>(NodeVector{hsigmoid}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, HSigmoidFusionWithReluDivF32) {
    {
        auto input = std::make_shared<opset7::Parameter>(element::f32, Shape{});
        auto add_constant = opset7::Constant::create(element::f32, Shape{}, {3.0});
        auto add = std::make_shared<opset7::Add>(input, add_constant);
        auto relu = std::make_shared<opset7::Relu>(add);
        auto min_constant = opset7::Constant::create(element::f32, Shape{}, {6.0});
        auto min = std::make_shared<opset7::Minimum>(relu, min_constant);
        auto div_constant = opset7::Constant::create(element::f32, Shape{}, {6.0});
        auto div = std::make_shared<opset7::Divide>(min, div_constant);

        model = std::make_shared<ov::Model>(NodeVector{div}, ParameterVector{input});

        manager.register_pass<ov::pass::HSigmoidFusion>();
    }

    {
        auto input = std::make_shared<opset7::Parameter>(element::f32, Shape{});
        auto hsigmoid = std::make_shared<opset7::HSigmoid>(input);

        model_ref = std::make_shared<ov::Model>(NodeVector{hsigmoid}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, HSigmoidFusionWithReluMul) {
    {
        auto input = std::make_shared<opset7::Parameter>(element::f16, PartialShape::dynamic(1));
        auto add_constant = opset7::Constant::create(element::f16, Shape{}, {3.0});
        auto add = std::make_shared<opset7::Add>(input, add_constant);
        auto relu = std::make_shared<opset7::Relu>(add);
        auto min_constant = opset7::Constant::create(element::f16, Shape{}, {6.0});
        auto min = std::make_shared<opset7::Minimum>(relu, min_constant);
        auto mul_constant = opset7::Constant::create(element::f16, Shape{}, {0.1666666716});
        auto mul_second = std::make_shared<opset7::Multiply>(min, mul_constant);

        model = std::make_shared<ov::Model>(NodeVector{mul_second}, ParameterVector{input});

        manager.register_pass<ov::pass::HSigmoidFusion>();
    }

    {
        auto input = std::make_shared<opset7::Parameter>(element::f16, PartialShape::dynamic(1));
        auto hsigmoid = std::make_shared<opset7::HSigmoid>(input);

        model_ref = std::make_shared<ov::Model>(NodeVector{hsigmoid}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, HSigmoidFusionWithoutRelu) {
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

        model = std::make_shared<ov::Model>(NodeVector{div}, ParameterVector{input});

        manager.register_pass<ov::pass::HSigmoidFusion>();
    }

    {
        auto input = std::make_shared<opset7::Parameter>(element::f16, PartialShape::dynamic(1));
        auto hsigmoid = std::make_shared<opset7::HSigmoid>(input);

        model_ref = std::make_shared<ov::Model>(NodeVector{hsigmoid}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, HSigmoidFusionWithClampMul) {
    {
        auto input = std::make_shared<opset7::Parameter>(element::f16, PartialShape::dynamic(1));
        auto add_constant = opset7::Constant::create(element::f16, Shape{}, {3.0});
        auto add = std::make_shared<opset7::Add>(input, add_constant);
        auto clamp = std::make_shared<op::v0::Clamp>(add, 0.0f, 6.0f);
        auto mul_constant = opset7::Constant::create(element::f16, Shape{}, {1.0 / 6.0});
        auto mul_first = std::make_shared<opset7::Multiply>(clamp, mul_constant);

        model = std::make_shared<ov::Model>(NodeVector{mul_first}, ParameterVector{input});

        manager.register_pass<ov::pass::HSigmoidFusion>();
    }

    {
        auto input = std::make_shared<opset7::Parameter>(element::f16, PartialShape::dynamic(1));
        auto hsigmoid = std::make_shared<opset7::HSigmoid>(input);

        model_ref = std::make_shared<ov::Model>(NodeVector{hsigmoid}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, HSigmoidFusionWithClampDiv) {
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(element::f16, PartialShape::dynamic(1));
        auto add_constant = ov::op::v0::Constant::create(element::f16, Shape{}, {3.0});
        auto add = std::make_shared<ov::op::v1::Add>(input, add_constant);
        auto clamp = std::make_shared<ov::op::v0::Clamp>(add, 0.0f, 6.0f);
        auto div_constant = ov::op::v0::Constant::create(element::f16, Shape{}, {6.0});
        auto div = std::make_shared<ov::op::v1::Divide>(clamp, div_constant);

        model = std::make_shared<ov::Model>(NodeVector{div}, ParameterVector{input});

        manager.register_pass<ov::pass::HSigmoidFusion>();
    }

    {
        auto input = std::make_shared<opset7::Parameter>(element::f16, PartialShape::dynamic(1));
        auto hsigmoid = std::make_shared<opset7::HSigmoid>(input);

        model_ref = std::make_shared<ov::Model>(NodeVector{hsigmoid}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, HSigmoidFusionWithReluMulWrongConstValue) {
    {
        auto input = std::make_shared<opset7::Parameter>(element::f16, PartialShape::dynamic(1));
        auto add_constant = opset7::Constant::create(element::f16, Shape{}, {3.0});
        auto add = std::make_shared<opset7::Add>(input, add_constant);
        auto relu = std::make_shared<opset7::Relu>(add);
        auto min_constant = opset7::Constant::create(element::f16, Shape{}, {6.0});
        auto min = std::make_shared<opset7::Minimum>(relu, min_constant);
        auto mul_constant = opset7::Constant::create(element::f16, Shape{}, {0.167});
        auto mul_second = std::make_shared<opset7::Multiply>(min, mul_constant);

        model = std::make_shared<ov::Model>(NodeVector{mul_second}, ParameterVector{input});

        manager.register_pass<ov::pass::HSigmoidFusion>();
    }

    {
        auto input = std::make_shared<opset7::Parameter>(element::f16, PartialShape::dynamic(1));
        auto add_constant = opset7::Constant::create(element::f16, Shape{}, {3.0});
        auto add = std::make_shared<opset7::Add>(input, add_constant);
        auto relu = std::make_shared<opset7::Relu>(add);
        auto min_constant = opset7::Constant::create(element::f16, Shape{}, {6.0});
        auto min = std::make_shared<opset7::Minimum>(relu, min_constant);
        auto mul_constant = opset7::Constant::create(element::f16, Shape{}, {0.167});
        auto mul_second = std::make_shared<opset7::Multiply>(min, mul_constant);

        model_ref = std::make_shared<ov::Model>(NodeVector{mul_second}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, HSigmoidFusionWithReluDivWrongConstValue) {
    {
        auto input = std::make_shared<opset7::Parameter>(element::f16, Shape{});
        auto add_constant = opset7::Constant::create(element::f16, Shape{}, {3.01});
        auto add = std::make_shared<opset7::Add>(input, add_constant);
        auto relu = std::make_shared<opset7::Relu>(add);
        auto min_constant = opset7::Constant::create(element::f16, Shape{}, {6.002});
        auto min = std::make_shared<opset7::Minimum>(relu, min_constant);
        auto div_constant = opset7::Constant::create(element::f16, Shape{}, {0.0});
        auto div = std::make_shared<opset7::Divide>(min, div_constant);

        model = std::make_shared<ov::Model>(NodeVector{div}, ParameterVector{input});

        manager.register_pass<ov::pass::HSigmoidFusion>();
    }

    {
        auto input = std::make_shared<opset7::Parameter>(element::f16, Shape{});
        auto add_constant = opset7::Constant::create(element::f16, Shape{}, {3.01});
        auto add = std::make_shared<opset7::Add>(input, add_constant);
        auto relu = std::make_shared<opset7::Relu>(add);
        auto min_constant = opset7::Constant::create(element::f16, Shape{}, {6.002});
        auto min = std::make_shared<opset7::Minimum>(relu, min_constant);
        auto div_constant = opset7::Constant::create(element::f16, Shape{}, {0.0});
        auto div = std::make_shared<opset7::Divide>(min, div_constant);

        model_ref = std::make_shared<ov::Model>(NodeVector{div}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, HSigmoidFusionWithoutReluWrongConstValue) {
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

        model = std::make_shared<ov::Model>(NodeVector{div}, ParameterVector{input});

        manager.register_pass<ov::pass::HSigmoidFusion>();
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

        model_ref = std::make_shared<ov::Model>(NodeVector{div}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, HSigmoidFusionWithClampWrongConstValue) {
    {
        auto input = std::make_shared<opset7::Parameter>(element::f16, PartialShape::dynamic(1));
        auto add_constant = opset7::Constant::create(element::f16, Shape{}, {3.11});
        auto add = std::make_shared<opset7::Add>(input, add_constant);
        auto clamp = std::make_shared<op::v0::Clamp>(add, 0.11f, 6.02f);
        auto mul_constant = opset7::Constant::create(element::f16, Shape{}, {0.98 / 6.15});
        auto mul_first = std::make_shared<opset7::Multiply>(clamp, mul_constant);

        model = std::make_shared<ov::Model>(NodeVector{mul_first}, ParameterVector{input});

        manager.register_pass<ov::pass::HSigmoidFusion>();
    }

    {
        auto input = std::make_shared<opset7::Parameter>(element::f16, PartialShape::dynamic(1));
        auto add_constant = opset7::Constant::create(element::f16, Shape{}, {3.11});
        auto add = std::make_shared<opset7::Add>(input, add_constant);
        auto clamp = std::make_shared<op::v0::Clamp>(add, 0.11f, 6.02f);
        auto mul_constant = opset7::Constant::create(element::f16, Shape{}, {0.98 / 6.15});
        auto mul_first = std::make_shared<opset7::Multiply>(clamp, mul_constant);

        model_ref = std::make_shared<ov::Model>(NodeVector{mul_first}, ParameterVector{input});
    }
}
