// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/swish_fusion.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset4.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"
using namespace ov;
using namespace testing;

TEST_F(TransformationTestsF, SwishFusionWithBeta) {
    {
        auto input = std::make_shared<opset4::Parameter>(element::f32, PartialShape::dynamic(1));
        auto beta = std::make_shared<opset4::Parameter>(element::f32, Shape{});
        auto mul = std::make_shared<opset4::Multiply>(input, beta);
        auto neg = std::make_shared<opset4::Negative>(mul);
        auto exp = std::make_shared<opset4::Exp>(neg);
        auto constant = opset4::Constant::create(element::f32, Shape{1}, {1.0});
        auto add = std::make_shared<opset4::Add>(exp, constant);
        auto div = std::make_shared<opset4::Divide>(input, add);

        model = std::make_shared<ov::Model>(NodeVector{div}, ParameterVector{input, beta});

        manager.register_pass<ov::pass::SwishFusion>();
    }

    {
        auto input = std::make_shared<opset4::Parameter>(element::f32, PartialShape::dynamic(1));
        auto beta = std::make_shared<opset4::Parameter>(element::f32, Shape{});
        auto swish = std::make_shared<opset4::Swish>(input, beta);

        model_ref = std::make_shared<ov::Model>(NodeVector{swish}, ParameterVector{input, beta});
    }
}

TEST_F(TransformationTestsF, SwishFusionWithoutBeta) {
    {
        auto input = std::make_shared<opset4::Parameter>(element::f16, PartialShape::dynamic(1));
        auto neg = std::make_shared<opset4::Negative>(input);
        auto exp = std::make_shared<opset4::Exp>(neg);
        auto constant = opset4::Constant::create(element::f16, Shape{}, {1.0});
        auto add = std::make_shared<opset4::Add>(exp, constant);
        auto div = std::make_shared<opset4::Divide>(input, add);

        model = std::make_shared<ov::Model>(NodeVector{div}, ParameterVector{input});

        manager.register_pass<ov::pass::SwishFusion>();
    }

    {
        auto input = std::make_shared<opset4::Parameter>(element::f16, PartialShape::dynamic(1));
        auto swish = std::make_shared<opset4::Swish>(input);

        model_ref = std::make_shared<ov::Model>(NodeVector{swish}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, SwishFusionWithoutBetaNonOneAddConstant) {
    {
        auto input = std::make_shared<opset4::Parameter>(element::f16, PartialShape::dynamic(1));
        auto neg = std::make_shared<opset4::Negative>(input);
        auto exp = std::make_shared<opset4::Exp>(neg);
        auto constant = opset4::Constant::create(element::f16, Shape{}, {1.1});
        auto add = std::make_shared<opset4::Add>(exp, constant);
        auto div = std::make_shared<opset4::Divide>(input, add);

        model = std::make_shared<ov::Model>(NodeVector{div}, ParameterVector{input});

        manager.register_pass<ov::pass::SwishFusion>();
    }

    {
        auto input = std::make_shared<opset4::Parameter>(element::f16, PartialShape::dynamic(1));
        auto neg = std::make_shared<opset4::Negative>(input);
        auto exp = std::make_shared<opset4::Exp>(neg);
        auto constant = opset4::Constant::create(element::f16, Shape{}, {1.1});
        auto add = std::make_shared<opset4::Add>(exp, constant);
        auto div = std::make_shared<opset4::Divide>(input, add);

        model = std::make_shared<ov::Model>(NodeVector{div}, ParameterVector{input});

        model_ref = std::make_shared<ov::Model>(NodeVector{div}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, SwishFusionWithSigmoid) {
    {
        auto input = std::make_shared<opset4::Parameter>(element::f16, PartialShape::dynamic(1));
        auto sig = std::make_shared<opset4::Sigmoid>(input);
        auto mul = std::make_shared<opset4::Multiply>(input, sig);

        model = std::make_shared<ov::Model>(NodeVector{mul}, ParameterVector{input});

        manager.register_pass<ov::pass::SwishFusion>();
    }

    {
        auto input = std::make_shared<opset4::Parameter>(element::f16, PartialShape::dynamic(1));
        auto swish = std::make_shared<opset4::Swish>(input);

        model_ref = std::make_shared<ov::Model>(NodeVector{swish}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, SwishFusionWithSigmoidWithBeta) {
    {
        auto input = std::make_shared<opset4::Parameter>(element::f16, PartialShape::dynamic(1));
        auto beta = std::make_shared<opset4::Parameter>(element::f16, Shape{});
        auto mul_beta = std::make_shared<opset4::Multiply>(input, beta);
        auto sig = std::make_shared<opset4::Sigmoid>(mul_beta);
        auto mul = std::make_shared<opset4::Multiply>(input, sig);

        model = std::make_shared<ov::Model>(NodeVector{mul}, ParameterVector{input, beta});

        manager.register_pass<ov::pass::SwishFusion>();
    }

    {
        auto input = std::make_shared<opset4::Parameter>(element::f16, PartialShape::dynamic(1));
        auto beta = std::make_shared<opset4::Parameter>(element::f16, Shape{});
        auto swish = std::make_shared<opset4::Swish>(input, beta);

        model_ref = std::make_shared<ov::Model>(NodeVector{swish}, ParameterVector{input, beta});
    }
}

TEST_F(TransformationTestsF, SwishFusionWithSigmoidWithBetaConstant) {
    // test where the beta constant has multiple but the same value
    {
        auto input = std::make_shared<opset4::Parameter>(element::f16, PartialShape::dynamic(1));
        auto beta = opset4::Constant::create(element::f16, Shape{3}, {2.0, 2.0, 2.0});
        auto mul_beta = std::make_shared<opset4::Multiply>(input, beta);
        auto sig = std::make_shared<opset4::Sigmoid>(mul_beta);
        auto mul = std::make_shared<opset4::Multiply>(input, sig);

        model = std::make_shared<ov::Model>(NodeVector{mul}, ParameterVector{input});

        manager.register_pass<ov::pass::SwishFusion>();
    }

    {
        auto input = std::make_shared<opset4::Parameter>(element::f16, PartialShape::dynamic(1));
        auto beta = opset4::Constant::create(element::f16, Shape{}, {2.0});
        auto swish = std::make_shared<opset4::Swish>(input, beta);

        model_ref = std::make_shared<ov::Model>(NodeVector{swish}, ParameterVector{input});
    }
}
