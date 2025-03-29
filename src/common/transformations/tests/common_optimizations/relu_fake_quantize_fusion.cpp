// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/relu_fake_quantize_fusion.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <queue>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset5.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"

using namespace testing;
using namespace ov;

TEST_F(TransformationTestsF, ReluFakeQuantizeFusion) {
    Shape data_shape{1, 3, 14, 14};
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
        auto relu = std::make_shared<opset5::Relu>(data);
        auto input_low = opset5::Constant::create(element::f32, Shape{1, 3, 1, 1}, {0, 0, 0});
        auto input_high = opset5::Constant::create(element::f32, Shape{1, 3, 1, 1}, {20, 20, 20});
        auto output_low = opset5::Constant::create(element::f32, Shape{}, {0});
        auto output_high = opset5::Constant::create(element::f32, Shape{}, {10});
        auto fq = std::make_shared<opset5::FakeQuantize>(relu, input_low, input_high, output_low, output_high, 11);
        model = std::make_shared<Model>(NodeVector{fq}, ParameterVector{data});
        manager.register_pass<ov::pass::ReluFakeQuantizeFusion>();
    }
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
        auto input_low = opset5::Constant::create(element::f32, Shape{1, 3, 1, 1}, {0, 0, 0});
        auto input_high = opset5::Constant::create(element::f32, Shape{1, 3, 1, 1}, {20, 20, 20});
        auto output_low = opset5::Constant::create(element::f32, Shape{}, {0});
        auto output_high = opset5::Constant::create(element::f32, Shape{}, {10});
        auto fq = std::make_shared<opset5::FakeQuantize>(data, input_low, input_high, output_low, output_high, 11);
        model_ref = std::make_shared<Model>(NodeVector{fq}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, ReluFakeQuantizeFusionNegativeInputLow) {
    Shape data_shape{1, 3, 14, 14};
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
        auto relu = std::make_shared<opset5::Relu>(data);
        auto input_low = opset5::Constant::create(element::f32, Shape{1, 3, 1, 1}, {2, -2, -2});
        auto input_high = opset5::Constant::create(element::f32, Shape{1, 3, 1, 1}, {20, 20, 20});
        auto output_low = opset5::Constant::create(element::f32, Shape{}, {0});
        auto output_high = opset5::Constant::create(element::f32, Shape{}, {10});
        auto fq = std::make_shared<opset5::FakeQuantize>(relu, input_low, input_high, output_low, output_high, 11);
        model = std::make_shared<Model>(NodeVector{fq}, ParameterVector{data});
        manager.register_pass<ov::pass::ReluFakeQuantizeFusion>();
    }
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
        auto relu = std::make_shared<opset5::Relu>(data);
        auto input_low = opset5::Constant::create(element::f32, Shape{1, 3, 1, 1}, {2, -2, -2});
        auto input_high = opset5::Constant::create(element::f32, Shape{1, 3, 1, 1}, {20, 20, 20});
        auto output_low = opset5::Constant::create(element::f32, Shape{}, {0});
        auto output_high = opset5::Constant::create(element::f32, Shape{}, {10});
        auto fq = std::make_shared<opset5::FakeQuantize>(relu, input_low, input_high, output_low, output_high, 11);
        model_ref = std::make_shared<Model>(NodeVector{fq}, ParameterVector{data});
    }
}
