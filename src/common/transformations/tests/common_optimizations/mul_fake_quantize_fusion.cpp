// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/mul_fake_quantize_fusion.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <queue>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset5.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"

using namespace testing;
using namespace ov;

TEST_F(TransformationTestsF, MulFakeQuantizeFusionPositiveConstant) {
    Shape data_shape{1, 3, 14, 14};
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
        auto mul_const = opset5::Constant::create(element::f32, Shape{1}, {2});
        auto mul = std::make_shared<opset5::Multiply>(data, mul_const);
        auto input_low = opset5::Constant::create(element::f32, Shape{1}, {1});
        auto input_high = opset5::Constant::create(element::f32, Shape{1}, {20});
        auto output_low = opset5::Constant::create(element::f32, Shape{1, 3, 1, 1}, {0, 0, 0});
        auto output_high = opset5::Constant::create(element::f32, Shape{1}, {10});
        auto fq = std::make_shared<opset5::FakeQuantize>(mul, input_low, input_high, output_low, output_high, 11);
        model = std::make_shared<Model>(NodeVector{fq}, ParameterVector{data});
        manager.register_pass<ov::pass::MulFakeQuantizeFusion>();
    }
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
        auto input_low = opset5::Constant::create(element::f32, Shape{1}, {0.5});
        auto input_high = opset5::Constant::create(element::f32, Shape{1}, {10});
        auto output_low = opset5::Constant::create(element::f32, Shape{1, 3, 1, 1}, {0, 0, 0});
        auto output_high = opset5::Constant::create(element::f32, Shape{1}, {10});
        auto fq = std::make_shared<opset5::FakeQuantize>(data, input_low, input_high, output_low, output_high, 11);
        model_ref = std::make_shared<Model>(NodeVector{fq}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, MulFakeQuantizeFusionConstantOnFirstInput) {
    Shape data_shape{1, 3, 14, 14};
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
        auto mul_const = opset5::Constant::create(element::f32, Shape{1}, {2});
        auto mul = std::make_shared<opset5::Multiply>(mul_const, data);
        auto input_low = opset5::Constant::create(element::f32, Shape{1}, {1});
        auto input_high = opset5::Constant::create(element::f32, Shape{1}, {20});
        auto output_low = opset5::Constant::create(element::f32, Shape{1, 3, 1, 1}, {0, 0, 0});
        auto output_high = opset5::Constant::create(element::f32, Shape{1}, {10});
        auto fq = std::make_shared<opset5::FakeQuantize>(mul, input_low, input_high, output_low, output_high, 11);
        model = std::make_shared<Model>(NodeVector{fq}, ParameterVector{data});
        manager.register_pass<ov::pass::MulFakeQuantizeFusion>();
    }
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
        auto input_low = opset5::Constant::create(element::f32, Shape{1}, {0.5});
        auto input_high = opset5::Constant::create(element::f32, Shape{1}, {10});
        auto output_low = opset5::Constant::create(element::f32, Shape{1, 3, 1, 1}, {0, 0, 0});
        auto output_high = opset5::Constant::create(element::f32, Shape{1}, {10});
        auto fq = std::make_shared<opset5::FakeQuantize>(data, input_low, input_high, output_low, output_high, 11);
        model_ref = std::make_shared<Model>(NodeVector{fq}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, MulFakeQuantizeFusionReshape) {
    Shape data_shape{1, 3, 14, 14};
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
        auto mul_const = opset5::Constant::create(element::f32, Shape{3, 1, 1}, {2, 4, 5});
        auto mul = std::make_shared<opset5::Multiply>(data, mul_const);
        auto input_low = opset5::Constant::create(element::f32, Shape{1}, {1});
        auto input_high = opset5::Constant::create(element::f32, Shape{1}, {20});
        auto output_low = opset5::Constant::create(element::f32, Shape{}, {0});
        auto output_high = opset5::Constant::create(element::f32, Shape{}, {10});
        auto fq = std::make_shared<opset5::FakeQuantize>(mul, input_low, input_high, output_low, output_high, 11);
        model = std::make_shared<Model>(NodeVector{fq}, ParameterVector{data});
        manager.register_pass<ov::pass::MulFakeQuantizeFusion>();
    }
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
        auto input_low = opset5::Constant::create(element::f32, Shape{1, 3, 1, 1}, {0.5, 0.25, 0.2});
        auto input_high = opset5::Constant::create(element::f32, Shape{1, 3, 1, 1}, {10, 5, 4});
        auto output_low = opset5::Constant::create(element::f32, Shape{}, {0});
        auto output_high = opset5::Constant::create(element::f32, Shape{}, {10});
        auto fq = std::make_shared<opset5::FakeQuantize>(data, input_low, input_high, output_low, output_high, 11);
        model_ref = std::make_shared<Model>(NodeVector{fq}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, MulFakeQuantizeFusionConstantNonScalarWithEqualValues) {
    Shape data_shape{1, 3, 14, 14};
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
        auto mul_const = opset5::Constant::create(element::f32, Shape{1, 3, 1, 1}, {2, 2, 2});
        auto mul = std::make_shared<opset5::Multiply>(data, mul_const);
        auto input_low = opset5::Constant::create(element::f32, Shape{1}, {1});
        auto input_high = opset5::Constant::create(element::f32, Shape{1}, {20});
        auto output_low = opset5::Constant::create(element::f32, Shape{1, 3, 1, 1}, {-10, -10, -10});
        auto output_high = opset5::Constant::create(element::f32, Shape{1}, {10});
        auto fq = std::make_shared<opset5::FakeQuantize>(mul, input_low, input_high, output_low, output_high, 11);
        model = std::make_shared<Model>(NodeVector{fq}, ParameterVector{data});
        manager.register_pass<ov::pass::MulFakeQuantizeFusion>();
    }
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
        auto input_low = opset5::Constant::create(element::f32, Shape{1}, {0.5});
        auto input_high = opset5::Constant::create(element::f32, Shape{1}, {10});
        auto output_low = opset5::Constant::create(element::f32, Shape{1, 3, 1, 1}, {-10, -10, -10});
        auto output_high = opset5::Constant::create(element::f32, Shape{1}, {10});
        auto fq = std::make_shared<opset5::FakeQuantize>(data, input_low, input_high, output_low, output_high, 11);
        model_ref = std::make_shared<Model>(NodeVector{fq}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, MulFakeQuantizeFusionWithPerChannelConstant) {
    Shape data_shape{1, 3};
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
        auto mul_const = opset5::Constant::create(element::f32, Shape{3}, {2, 4, 5});
        auto mul = std::make_shared<opset5::Multiply>(data, mul_const);
        auto input_low = opset5::Constant::create(element::f32, Shape{1}, {1});
        auto input_high = opset5::Constant::create(element::f32, Shape{1}, {20});
        auto output_low = opset5::Constant::create(element::f32, Shape{}, {0});
        auto output_high = opset5::Constant::create(element::f32, Shape{}, {10});
        auto fq = std::make_shared<opset5::FakeQuantize>(mul, input_low, input_high, output_low, output_high, 11);
        model = std::make_shared<Model>(NodeVector{fq}, ParameterVector{data});
        manager.register_pass<ov::pass::MulFakeQuantizeFusion>();
    }
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
        auto input_low = opset5::Constant::create(element::f32, Shape{1, 3}, {0.5, 0.25, 0.2});
        auto input_high = opset5::Constant::create(element::f32, Shape{1, 3}, {10, 5, 4});
        auto output_low = opset5::Constant::create(element::f32, Shape{}, {0});
        auto output_high = opset5::Constant::create(element::f32, Shape{}, {10});
        auto fq = std::make_shared<opset5::FakeQuantize>(data, input_low, input_high, output_low, output_high, 11);
        model_ref = std::make_shared<Model>(NodeVector{fq}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, NegativeMulFakeQuantizeFusionNotAConstant) {
    Shape data_shape{1, 3, 14, 14};
    auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
    auto mul_2nd_input = std::make_shared<opset5::Parameter>(element::f32, Shape{1});
    auto mul = std::make_shared<opset5::Multiply>(data, mul_2nd_input);
    auto input_low = opset5::Constant::create(element::f32, Shape{1}, {0});
    auto input_high = opset5::Constant::create(element::f32, Shape{1}, {20});
    auto output_low = opset5::Constant::create(element::f32, Shape{}, {0});
    auto output_high = opset5::Constant::create(element::f32, Shape{}, {10});
    auto fq = std::make_shared<opset5::FakeQuantize>(mul, input_low, input_high, output_low, output_high, 11);
    model = std::make_shared<Model>(NodeVector{fq}, ParameterVector{data, mul_2nd_input});
    manager.register_pass<ov::pass::MulFakeQuantizeFusion>();
}

TEST_F(TransformationTestsF, NegativeMulFakeQuantizeFusionLowPrecision) {
    Shape data_shape{1, 3, 14, 14};
    auto data = std::make_shared<opset5::Parameter>(element::f16, data_shape);
    auto mul_const = opset5::Constant::create(element::f16, Shape{1}, {2});
    auto mul = std::make_shared<opset5::Multiply>(data, mul_const);
    auto input_low = opset5::Constant::create(element::f16, Shape{1}, {1});
    auto input_high = opset5::Constant::create(element::f16, Shape{1}, {20});
    auto output_low = opset5::Constant::create(element::f16, Shape{1, 3, 1, 1}, {0, 0, 0});
    auto output_high = opset5::Constant::create(element::f16, Shape{1}, {10});
    auto fq = std::make_shared<opset5::FakeQuantize>(mul, input_low, input_high, output_low, output_high, 11);
    model = std::make_shared<Model>(NodeVector{fq}, ParameterVector{data});
    model_ref = model->clone();
    manager.register_pass<ov::pass::MulFakeQuantizeFusion>();
}

TEST_F(TransformationTestsF, NegativeMulFakeQuantizeFusionConstantAllNegative) {
    Shape data_shape{1, 3, 14, 14};
    auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
    auto mul_const = opset5::Constant::create(element::f32, Shape{1}, {-2});
    auto mul = std::make_shared<opset5::Multiply>(data, mul_const);
    auto input_low = opset5::Constant::create(element::f32, Shape{1}, {1});
    auto input_high = opset5::Constant::create(element::f32, Shape{1}, {20});
    auto output_low = opset5::Constant::create(element::f32, Shape{1, 3, 1, 1}, {-10, -10, -10});
    auto output_high = opset5::Constant::create(element::f32, Shape{1}, {10});
    auto fq = std::make_shared<opset5::FakeQuantize>(mul, input_low, input_high, output_low, output_high, 11);
    model = std::make_shared<Model>(NodeVector{fq}, ParameterVector{data});
    model_ref = model->clone();
    manager.register_pass<ov::pass::MulFakeQuantizeFusion>();
}

TEST_F(TransformationTestsF, NegativeMulFakeQuantizeFusionConstantSomeNegative) {
    Shape data_shape{1, 3, 14, 14};
    auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
    auto mul_const = opset5::Constant::create(element::f32, Shape{3, 1, 1}, {2, 1, -2});
    auto mul = std::make_shared<opset5::Multiply>(data, mul_const);
    auto input_low = opset5::Constant::create(element::f32, Shape{1}, {1});
    auto input_high = opset5::Constant::create(element::f32, Shape{1}, {20});
    auto output_low = opset5::Constant::create(element::f32, Shape{1, 3, 1, 1}, {-10, -10, -10});
    auto output_high = opset5::Constant::create(element::f32, Shape{1}, {10});
    auto fq = std::make_shared<opset5::FakeQuantize>(mul, input_low, input_high, output_low, output_high, 20);
    model = std::make_shared<Model>(NodeVector{fq}, ParameterVector{data});
    model_ref = model->clone();
    manager.register_pass<ov::pass::MulFakeQuantizeFusion>();
}

TEST_F(TransformationTestsF, NegativeMulFakeQuantizeFusionWithNonPerChannelConstant) {
    Shape data_shape{1, 4, 3};
    auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
    auto mul_const = opset5::Constant::create(element::f32, Shape{3}, {2, 3, 4});
    auto mul = std::make_shared<opset5::Multiply>(data, mul_const);
    auto input_low = opset5::Constant::create(element::f32, Shape{1}, {0});
    auto input_high = opset5::Constant::create(element::f32, Shape{1}, {20});
    auto output_low = opset5::Constant::create(element::f32, Shape{}, {0});
    auto output_high = opset5::Constant::create(element::f32, Shape{}, {10});
    auto fq = std::make_shared<opset5::FakeQuantize>(mul, input_low, input_high, output_low, output_high, 11);
    model = std::make_shared<Model>(NodeVector{fq}, ParameterVector{data});
    manager.register_pass<ov::pass::MulFakeQuantizeFusion>();
}

TEST_F(TransformationTestsF, MulFakeQuantizeFusionWithBroadcastingConstant) {
    auto data = std::make_shared<opset5::Parameter>(element::f32, ov::PartialShape{DYN, 3});
    auto mul_const = opset5::Constant::create(element::f32, Shape{3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto mul = std::make_shared<opset5::Multiply>(data, mul_const);
    auto input_low = opset5::Constant::create(element::f32, Shape{1}, {1});
    auto input_high = opset5::Constant::create(element::f32, Shape{1}, {20});
    auto output_low = opset5::Constant::create(element::f32, Shape{}, {0});
    auto output_high = opset5::Constant::create(element::f32, Shape{}, {10});
    auto fq = std::make_shared<opset5::FakeQuantize>(mul, input_low, input_high, output_low, output_high, 11);
    model = std::make_shared<Model>(NodeVector{fq}, ParameterVector{data});
    manager.register_pass<ov::pass::MulFakeQuantizeFusion>();
}
