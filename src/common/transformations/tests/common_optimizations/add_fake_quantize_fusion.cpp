// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/add_fake_quantize_fusion.hpp"

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

TEST_F(TransformationTestsF, AddFakeQuantizeFusion) {
    Shape data_shape{1, 3, 14, 14};
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
        auto add_const = opset5::Constant::create(element::f32, Shape{1}, {2});
        auto add = std::make_shared<opset5::Add>(data, add_const);
        auto input_low = opset5::Constant::create(element::f32, Shape{1}, {0});
        auto input_high = opset5::Constant::create(element::f32, Shape{1}, {20});
        auto output_low = opset5::Constant::create(element::f32, Shape{}, {0});
        auto output_high = opset5::Constant::create(element::f32, Shape{}, {10});
        auto fq = std::make_shared<opset5::FakeQuantize>(add, input_low, input_high, output_low, output_high, 11);
        model = std::make_shared<Model>(NodeVector{fq}, ParameterVector{data});
        manager.register_pass<ov::pass::AddFakeQuantizeFusion>();
    }
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
        auto input_low = opset5::Constant::create(element::f32, Shape{1}, {-2});
        auto input_high = opset5::Constant::create(element::f32, Shape{1}, {18});
        auto output_low = opset5::Constant::create(element::f32, Shape{}, {0});
        auto output_high = opset5::Constant::create(element::f32, Shape{}, {10});
        auto fq = std::make_shared<opset5::FakeQuantize>(data, input_low, input_high, output_low, output_high, 11);
        model_ref = std::make_shared<Model>(NodeVector{fq}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, AddFakeQuantizeFusionWithConvolutionAndScalarConstant) {
    Shape data_shape{1, 3, 14, 14};
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
        auto filter = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 3, 2, 2});
        auto conv = std::make_shared<opset5::Convolution>(data,
                                                          filter,
                                                          Strides{1, 1},
                                                          CoordinateDiff{0, 0},
                                                          CoordinateDiff{0, 0},
                                                          Strides{1, 1});
        auto add_const = opset5::Constant::create(element::f32, Shape{1}, {2});
        auto add = std::make_shared<opset5::Add>(conv, add_const);
        auto input_low = opset5::Constant::create(element::f32, Shape{1}, {0});
        auto input_high = opset5::Constant::create(element::f32, Shape{1}, {20});
        auto output_low = opset5::Constant::create(element::f32, Shape{}, {0});
        auto output_high = opset5::Constant::create(element::f32, Shape{}, {10});
        auto fq = std::make_shared<opset5::FakeQuantize>(add, input_low, input_high, output_low, output_high, 11);
        model = std::make_shared<Model>(NodeVector{fq}, ParameterVector{data, filter});
        manager.register_pass<ov::pass::AddFakeQuantizeFusion>();
    }
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
        auto filter = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 3, 2, 2});
        auto conv = std::make_shared<opset5::Convolution>(data,
                                                          filter,
                                                          Strides{1, 1},
                                                          CoordinateDiff{0, 0},
                                                          CoordinateDiff{0, 0},
                                                          Strides{1, 1});
        auto input_low = opset5::Constant::create(element::f32, Shape{1}, {-2});
        auto input_high = opset5::Constant::create(element::f32, Shape{1}, {18});
        auto output_low = opset5::Constant::create(element::f32, Shape{}, {0});
        auto output_high = opset5::Constant::create(element::f32, Shape{}, {10});
        auto fq = std::make_shared<opset5::FakeQuantize>(conv, input_low, input_high, output_low, output_high, 11);
        model_ref = std::make_shared<Model>(NodeVector{fq}, ParameterVector{data, filter});
    }
}

TEST_F(TransformationTestsF, AddFakeQuantizeFusionConstantOnFirstInput) {
    Shape data_shape{1, 3, 14, 14};
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
        auto add_const = opset5::Constant::create(element::f32, Shape{1}, {2});
        auto add = std::make_shared<opset5::Add>(add_const, data);
        auto input_low = opset5::Constant::create(element::f32, Shape{1}, {0});
        auto input_high = opset5::Constant::create(element::f32, Shape{1}, {20});
        auto output_low = opset5::Constant::create(element::f32, Shape{}, {0});
        auto output_high = opset5::Constant::create(element::f32, Shape{}, {10});
        auto fq = std::make_shared<opset5::FakeQuantize>(add, input_low, input_high, output_low, output_high, 11);
        model = std::make_shared<Model>(NodeVector{fq}, ParameterVector{data});
        manager.register_pass<ov::pass::AddFakeQuantizeFusion>();
    }
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
        auto input_low = opset5::Constant::create(element::f32, Shape{1}, {-2});
        auto input_high = opset5::Constant::create(element::f32, Shape{1}, {18});
        auto output_low = opset5::Constant::create(element::f32, Shape{}, {0});
        auto output_high = opset5::Constant::create(element::f32, Shape{}, {10});
        auto fq = std::make_shared<opset5::FakeQuantize>(data, input_low, input_high, output_low, output_high, 11);
        model_ref = std::make_shared<Model>(NodeVector{fq}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, AddFakeQuantizeFusionConstantWithEqualValues) {
    Shape data_shape{1, 3, 14, 14};
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
        auto add_const = opset5::Constant::create(element::f32, Shape{1, 3, 1, 1}, {2, 2, 2});
        auto add = std::make_shared<opset5::Add>(add_const, data);
        auto input_low = opset5::Constant::create(element::f32, Shape{1}, {0});
        auto input_high = opset5::Constant::create(element::f32, Shape{1}, {20});
        auto output_low = opset5::Constant::create(element::f32, Shape{}, {0});
        auto output_high = opset5::Constant::create(element::f32, Shape{}, {10});
        auto fq = std::make_shared<opset5::FakeQuantize>(add, input_low, input_high, output_low, output_high, 11);
        model = std::make_shared<Model>(NodeVector{fq}, ParameterVector{data});
        manager.register_pass<ov::pass::AddFakeQuantizeFusion>();
    }
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
        auto input_low = opset5::Constant::create(element::f32, Shape{1}, {-2});
        auto input_high = opset5::Constant::create(element::f32, Shape{1}, {18});
        auto output_low = opset5::Constant::create(element::f32, Shape{}, {0});
        auto output_high = opset5::Constant::create(element::f32, Shape{}, {10});
        auto fq = std::make_shared<opset5::FakeQuantize>(data, input_low, input_high, output_low, output_high, 11);
        model_ref = std::make_shared<Model>(NodeVector{fq}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, AddFakeQuantizeFusionReshape) {
    Shape data_shape{1, 3, 14, 14};
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
        auto add_const = opset5::Constant::create(element::f32, Shape{3, 1, 1}, {2, 3, 4});
        auto add = std::make_shared<opset5::Add>(data, add_const);
        auto input_low = opset5::Constant::create(element::f32, Shape{1}, {0});
        auto input_high = opset5::Constant::create(element::f32, Shape{1}, {20});
        auto output_low = opset5::Constant::create(element::f32, Shape{}, {0});
        auto output_high = opset5::Constant::create(element::f32, Shape{}, {10});
        auto fq = std::make_shared<opset5::FakeQuantize>(add, input_low, input_high, output_low, output_high, 11);
        model = std::make_shared<Model>(NodeVector{fq}, ParameterVector{data});
        manager.register_pass<ov::pass::AddFakeQuantizeFusion>();
    }
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
        auto input_low = opset5::Constant::create(element::f32, Shape{1, 3, 1, 1}, {-2, -3, -4});
        auto input_high = opset5::Constant::create(element::f32, Shape{1, 3, 1, 1}, {18, 17, 16});
        auto output_low = opset5::Constant::create(element::f32, Shape{}, {0});
        auto output_high = opset5::Constant::create(element::f32, Shape{}, {10});
        auto fq = std::make_shared<opset5::FakeQuantize>(data, input_low, input_high, output_low, output_high, 11);
        model_ref = std::make_shared<Model>(NodeVector{fq}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, AddFakeQuantizeFusionWithPerChannelConstant) {
    Shape data_shape{1, 3};
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
        auto add_const = opset5::Constant::create(element::f32, Shape{3}, {2, 3, 4});
        auto add = std::make_shared<opset5::Add>(data, add_const);
        auto input_low = opset5::Constant::create(element::f32, Shape{1}, {0});
        auto input_high = opset5::Constant::create(element::f32, Shape{1}, {20});
        auto output_low = opset5::Constant::create(element::f32, Shape{}, {0});
        auto output_high = opset5::Constant::create(element::f32, Shape{}, {10});
        auto fq = std::make_shared<opset5::FakeQuantize>(add, input_low, input_high, output_low, output_high, 11);
        model = std::make_shared<Model>(NodeVector{fq}, ParameterVector{data});
        manager.register_pass<ov::pass::AddFakeQuantizeFusion>();
    }
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
        auto input_low = opset5::Constant::create(element::f32, Shape{1, 3}, {-2, -3, -4});
        auto input_high = opset5::Constant::create(element::f32, Shape{1, 3}, {18, 17, 16});
        auto output_low = opset5::Constant::create(element::f32, Shape{}, {0});
        auto output_high = opset5::Constant::create(element::f32, Shape{}, {10});
        auto fq = std::make_shared<opset5::FakeQuantize>(data, input_low, input_high, output_low, output_high, 11);
        model_ref = std::make_shared<Model>(NodeVector{fq}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, NegativeAddFakeQuantizeFusionNotAConstant) {
    Shape data_shape{1, 3, 14, 14};
    auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
    auto add_2nd_input = std::make_shared<opset5::Parameter>(element::f32, Shape{1});
    auto add = std::make_shared<opset5::Add>(data, add_2nd_input);
    auto input_low = opset5::Constant::create(element::f32, Shape{1}, {0});
    auto input_high = opset5::Constant::create(element::f32, Shape{1}, {20});
    auto output_low = opset5::Constant::create(element::f32, Shape{}, {0});
    auto output_high = opset5::Constant::create(element::f32, Shape{}, {10});
    auto fq = std::make_shared<opset5::FakeQuantize>(add, input_low, input_high, output_low, output_high, 11);
    model = std::make_shared<Model>(NodeVector{fq}, ParameterVector{data, add_2nd_input});
    manager.register_pass<ov::pass::AddFakeQuantizeFusion>();
}

TEST_F(TransformationTestsF, NegativeAddFakeQuantizeFusionWithConvolutionAndNonScalarConstant) {
    Shape data_shape{1, 3, 14, 14};
    auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
    auto filter = std::make_shared<opset5::Parameter>(element::f32, Shape{4, 3, 2, 2});
    auto conv = std::make_shared<opset5::Convolution>(data,
                                                      filter,
                                                      Strides{1, 1},
                                                      CoordinateDiff{0, 0},
                                                      CoordinateDiff{0, 0},
                                                      Strides{1, 1});
    auto add_const = opset5::Constant::create(element::f32, Shape{1, 4, 1, 1}, {1, 2, 3, 4});
    auto add = std::make_shared<opset5::Add>(conv, add_const);
    auto input_low = opset5::Constant::create(element::f32, Shape{1}, {0});
    auto input_high = opset5::Constant::create(element::f32, Shape{1}, {20});
    auto output_low = opset5::Constant::create(element::f32, Shape{}, {0});
    auto output_high = opset5::Constant::create(element::f32, Shape{}, {10});
    auto fq = std::make_shared<opset5::FakeQuantize>(add, input_low, input_high, output_low, output_high, 11);
    model = std::make_shared<Model>(NodeVector{fq}, ParameterVector{data, filter});
    manager.register_pass<ov::pass::AddFakeQuantizeFusion>();
}

TEST_F(TransformationTestsF, NegativeAddFakeQuantizeFusionLowPrecision) {
    Shape data_shape{1, 3, 14, 14};
    auto data = std::make_shared<opset5::Parameter>(element::f16, data_shape);
    auto add_const = opset5::Constant::create(element::f16, Shape{1}, {2});
    auto add = std::make_shared<opset5::Add>(data, add_const);
    auto input_low = opset5::Constant::create(element::f16, Shape{1}, {0});
    auto input_high = opset5::Constant::create(element::f16, Shape{1}, {20});
    auto output_low = opset5::Constant::create(element::f16, Shape{}, {0});
    auto output_high = opset5::Constant::create(element::f16, Shape{}, {10});
    auto fq = std::make_shared<opset5::FakeQuantize>(add, input_low, input_high, output_low, output_high, 11);
    model = std::make_shared<Model>(NodeVector{fq}, ParameterVector{data});
    model_ref = model->clone();
    manager.register_pass<ov::pass::AddFakeQuantizeFusion>();
}

TEST_F(TransformationTestsF, NegativeAddFakeQuantizeFusionWithNonPerChannelConstant) {
    Shape data_shape{1, 4, 3};
    auto data = std::make_shared<opset5::Parameter>(element::f32, data_shape);
    auto add_const = opset5::Constant::create(element::f32, Shape{3}, {2, 3, 4});
    auto add = std::make_shared<opset5::Add>(data, add_const);
    auto input_low = opset5::Constant::create(element::f32, Shape{1}, {0});
    auto input_high = opset5::Constant::create(element::f32, Shape{1}, {20});
    auto output_low = opset5::Constant::create(element::f32, Shape{}, {0});
    auto output_high = opset5::Constant::create(element::f32, Shape{}, {10});
    auto fq = std::make_shared<opset5::FakeQuantize>(add, input_low, input_high, output_low, output_high, 11);
    model = std::make_shared<Model>(NodeVector{fq}, ParameterVector{data});
    manager.register_pass<ov::pass::AddFakeQuantizeFusion>();
}

TEST_F(TransformationTestsF, AddFakeQuantizeFusionWithBroadcastingConstant) {
    auto data = std::make_shared<opset5::Parameter>(element::f32, ov::PartialShape{DYN, 3});
    auto add_const = opset5::Constant::create(element::f32, Shape{3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto add = std::make_shared<opset5::Add>(data, add_const);
    auto input_low = opset5::Constant::create(element::f32, Shape{1}, {0});
    auto input_high = opset5::Constant::create(element::f32, Shape{1}, {20});
    auto output_low = opset5::Constant::create(element::f32, Shape{}, {0});
    auto output_high = opset5::Constant::create(element::f32, Shape{}, {10});
    auto fq = std::make_shared<opset5::FakeQuantize>(add, input_low, input_high, output_low, output_high, 11);
    model = std::make_shared<Model>(NodeVector{fq}, ParameterVector{data});
    manager.register_pass<ov::pass::AddFakeQuantizeFusion>();
}
