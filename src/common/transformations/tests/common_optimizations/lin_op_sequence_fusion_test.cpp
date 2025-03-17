// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/lin_op_sequence_fusion.hpp"

#include <gtest/gtest.h>

#include <fstream>
#include <map>
#include <memory>
#include <queue>
#include <sstream>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "common_test_utils/test_common.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset3.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/visualize_tree.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"

using namespace testing;
using namespace ov;

TEST_F(TransformationTestsF, MulAddMulAddFusion) {
    {
        auto input = std::make_shared<opset3::Parameter>(element::f32, Shape{1, 128, 3072});
        auto mul1_const = opset3::Constant::create(element::f32, Shape{128, 1}, {2});
        auto mul2_const = opset3::Constant::create(element::f32, Shape{128, 1}, {3});
        auto add1_const = opset3::Constant::create(element::f32, Shape{128, 1}, {4});
        auto add2_const = opset3::Constant::create(element::f32, Shape{128, 1}, {5});

        auto mul1 = std::make_shared<opset3::Multiply>(input, mul1_const);
        auto add1 = std::make_shared<opset3::Add>(mul1, add1_const);
        auto mul2 = std::make_shared<opset3::Multiply>(add1, mul2_const);
        auto add2 = std::make_shared<opset3::Add>(mul2, add2_const);

        model = std::make_shared<ov::Model>(NodeVector{add2}, ParameterVector{input});
        manager.register_pass<ov::pass::LinOpSequenceFusion>();
    }

    {
        auto input = std::make_shared<opset3::Parameter>(element::f32, Shape{1, 128, 3072});
        auto mul1_const = opset3::Constant::create(element::f32, Shape{128, 1}, {6});
        auto add1_const = opset3::Constant::create(element::f32, Shape{128, 1}, {17});

        auto mul1 = std::make_shared<opset3::Multiply>(input, mul1_const);
        auto add1 = std::make_shared<opset3::Add>(mul1, add1_const);

        model_ref = std::make_shared<ov::Model>(NodeVector{add1}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, MulMulMulFusion) {
    {
        auto input = std::make_shared<opset3::Parameter>(element::f32, Shape{1, 128, 3072});
        auto mul1_const = opset3::Constant::create(element::f32, Shape{128, 1}, {2});
        auto mul2_const = opset3::Constant::create(element::f32, Shape{128, 1}, {3});
        auto mul3_const = opset3::Constant::create(element::f32, Shape{1}, {3});

        auto mul1 = std::make_shared<opset3::Multiply>(input, mul1_const);
        auto mul2 = std::make_shared<opset3::Multiply>(mul1, mul2_const);
        auto mul3 = std::make_shared<opset3::Multiply>(mul2, mul3_const);

        model = std::make_shared<ov::Model>(NodeVector{mul2}, ParameterVector{input});
        manager.register_pass<ov::pass::LinOpSequenceFusion>();
    }

    {
        auto input = std::make_shared<opset3::Parameter>(element::f32, Shape{1, 128, 3072});
        auto mul1_const = opset3::Constant::create(element::f32, Shape{128, 1}, {12});

        auto mul1 = std::make_shared<opset3::Multiply>(input, mul1_const);

        model_ref = std::make_shared<ov::Model>(NodeVector{mul1}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, MulMulMulFusion_f64) {
    {
        auto input = std::make_shared<opset3::Parameter>(element::f64, Shape{1, 128, 3072});
        auto mul1_const = opset3::Constant::create(element::f64, Shape{128, 1}, {2});
        auto mul2_const = opset3::Constant::create(element::f64, Shape{128, 1}, {3});
        auto mul3_const = opset3::Constant::create(element::f64, Shape{1}, {3});

        auto mul1 = std::make_shared<opset3::Multiply>(input, mul1_const);
        auto mul2 = std::make_shared<opset3::Multiply>(mul1, mul2_const);
        auto mul3 = std::make_shared<opset3::Multiply>(mul2, mul3_const);

        model = std::make_shared<ov::Model>(NodeVector{mul2}, ParameterVector{input});
        manager.register_pass<ov::pass::LinOpSequenceFusion>();
    }

    {
        auto input = std::make_shared<opset3::Parameter>(element::f64, Shape{1, 128, 3072});
        auto mul1_const = opset3::Constant::create(element::f64, Shape{128, 1}, {12});

        auto mul1 = std::make_shared<opset3::Multiply>(input, mul1_const);

        model_ref = std::make_shared<ov::Model>(NodeVector{mul1}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, MulMulMulFusion_not_supported_type) {
    constexpr auto et = element::u8;
    {
        auto input = std::make_shared<opset3::Parameter>(et, Shape{1, 128, 3072});
        auto mul1_const = opset3::Constant::create(et, Shape{128, 1}, {2});
        auto mul2_const = opset3::Constant::create(et, Shape{128, 1}, {3});
        auto mul3_const = opset3::Constant::create(et, Shape{1}, {3});

        auto mul1 = std::make_shared<opset3::Multiply>(input, mul1_const);
        auto mul2 = std::make_shared<opset3::Multiply>(mul1, mul2_const);
        auto mul3 = std::make_shared<opset3::Multiply>(mul2, mul3_const);

        model = std::make_shared<ov::Model>(NodeVector{mul2}, ParameterVector{input});
        manager.register_pass<ov::pass::LinOpSequenceFusion>();
    }
}

TEST_F(TransformationTestsF, AddAddAddFusion) {
    {
        auto input = std::make_shared<opset3::Parameter>(element::f32, Shape{1, 128, 3072});
        auto add1_const = opset3::Constant::create(element::f32, Shape{128, 1}, {2});
        auto add2_const = opset3::Constant::create(element::f32, Shape{128, 1}, {3});
        auto add3_const = opset3::Constant::create(element::f32, Shape{1}, {3});

        auto add1 = std::make_shared<opset3::Add>(input, add1_const);
        auto add2 = std::make_shared<opset3::Add>(add1, add2_const);
        auto add3 = std::make_shared<opset3::Add>(add2, add3_const);

        model = std::make_shared<ov::Model>(NodeVector{add3}, ParameterVector{input});
        manager.register_pass<ov::pass::LinOpSequenceFusion>();
    }

    {
        auto input = std::make_shared<opset3::Parameter>(element::f32, Shape{1, 128, 3072});
        auto add1_const = opset3::Constant::create(element::f32, Shape{128, 1}, {8});

        auto add1 = std::make_shared<opset3::Add>(input, add1_const);

        model_ref = std::make_shared<ov::Model>(NodeVector{add1}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, MulAddAddMulFusion) {
    {
        auto input = std::make_shared<opset3::Parameter>(element::f32, Shape{1, 128, 3072});
        auto mul1_const = opset3::Constant::create(element::f32, Shape{128, 1}, {2});
        auto mul2_const = opset3::Constant::create(element::f32, Shape{128, 1}, {3});
        auto add1_const = opset3::Constant::create(element::f32, Shape{128, 1}, {4});
        auto add2_const = opset3::Constant::create(element::f32, Shape{128, 1}, {5});

        auto mul1 = std::make_shared<opset3::Multiply>(input, mul1_const);
        auto add1 = std::make_shared<opset3::Add>(mul1, add1_const);
        auto add2 = std::make_shared<opset3::Add>(add1, add2_const);
        auto mul2 = std::make_shared<opset3::Multiply>(add2, mul2_const);

        model = std::make_shared<ov::Model>(NodeVector{mul2}, ParameterVector{input});
        manager.register_pass<ov::pass::LinOpSequenceFusion>();
    }

    {
        auto input = std::make_shared<opset3::Parameter>(element::f32, Shape{1, 128, 3072});
        auto mul1_const = opset3::Constant::create(element::f32, Shape{128, 1}, {10});
        auto add1_const = opset3::Constant::create(element::f32, Shape{128, 1}, {40});

        auto mul1 = std::make_shared<opset3::Multiply>(input, mul1_const);
        auto add1 = std::make_shared<opset3::Add>(mul1, add1_const);

        model_ref = std::make_shared<ov::Model>(NodeVector{add1}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, AddAddAddFusionF64) {
    {
        auto input = std::make_shared<opset3::Parameter>(element::f64, Shape{1, 128, 3072});
        auto add1_const = opset3::Constant::create(element::f64, Shape{128, 1}, {2});
        auto add2_const = opset3::Constant::create(element::f64, Shape{128, 1}, {3});
        auto add3_const = opset3::Constant::create(element::f64, Shape{1}, {3});

        auto add1 = std::make_shared<opset3::Add>(input, add1_const);
        auto add2 = std::make_shared<opset3::Add>(add1, add2_const);
        auto add3 = std::make_shared<opset3::Add>(add2, add3_const);

        model = std::make_shared<ov::Model>(NodeVector{add3}, ParameterVector{input});
        manager.register_pass<ov::pass::LinOpSequenceFusion>();
    }
    {
        auto input = std::make_shared<opset3::Parameter>(element::f64, Shape{1, 128, 3072});
        auto add1_const = opset3::Constant::create(element::f64, Shape{128, 1}, {2});
        auto add2_const = opset3::Constant::create(element::f64, Shape{128, 1}, {3});
        auto add3_const = opset3::Constant::create(element::f64, Shape{1}, {3});

        auto add1 = std::make_shared<opset3::Add>(add1_const, add1_const);

        auto add2 = std::make_shared<opset3::Add>(input, add1);

        auto add3 = std::make_shared<opset3::Add>(add2, add3_const);

        model_ref = std::make_shared<ov::Model>(NodeVector{add3}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, MulAddAddMulFusionF64) {
    {
        auto input = std::make_shared<opset3::Parameter>(element::f64, Shape{1, 128, 3072});
        auto add1_const = opset3::Constant::create(element::f64, Shape{128, 1}, {4});
        auto add1 = std::make_shared<opset3::Add>(input, add1_const);
        auto mul1_const = opset3::Constant::create(element::f64, Shape{128, 1}, {2});
        auto mul1 = std::make_shared<opset3::Multiply>(add1, mul1_const);

        model = std::make_shared<ov::Model>(NodeVector{mul1}, ParameterVector{input});
        manager.register_pass<ov::pass::LinOpSequenceFusion>();
    }

    {
        auto input = std::make_shared<opset3::Parameter>(element::f64, Shape{1, 128, 3072});
        auto mul1_const = opset3::Constant::create(element::f64, Shape{128, 1}, {10});
        auto add1_const = opset3::Constant::create(element::f64, Shape{128, 1}, {40});

        auto mul1 = std::make_shared<opset3::Multiply>(input, mul1_const);
        auto add1 = std::make_shared<opset3::Add>(mul1, add1_const);

        model_ref = std::make_shared<ov::Model>(NodeVector{add1}, ParameterVector{input});
    }
}
