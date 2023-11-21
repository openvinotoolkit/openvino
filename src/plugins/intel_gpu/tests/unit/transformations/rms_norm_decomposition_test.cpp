// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <openvino/core/model.hpp>
#include <openvino/opsets/opset10.hpp>
#include <intel_gpu/op/rms.hpp>
#include <plugin/transformations/rms_fusion.hpp>
#include <transformations/utils/utils.hpp>
#include <openvino/pass/manager.hpp>

#include "common_test_utils/ov_test_utils.hpp"

using namespace testing;
using namespace ov::intel_gpu;

TEST_F(TransformationTestsF, RMSNormFusionTest1) {
    {
        auto input = std::make_shared<ov::opset10::Parameter>(ov::element::f32, ov::Shape{1, 2, 6});
        auto power_const = ov::opset10::Constant::create(ov::element::f32, {}, {2.f});
        auto power = std::make_shared<ov::opset10::Power>(input, power_const);
        auto mean_axes = ov::opset10::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
        auto mean = std::make_shared<ov::opset10::ReduceMean>(power, mean_axes, true);
        auto eps = ov::opset10::Constant::create(ov::element::f32, {}, {1e-5f});
        auto add_eps = std::make_shared<ov::opset10::Add>(mean, eps);
        auto sqrt = std::make_shared<ov::opset10::Sqrt>(add_eps);
        auto div_const = ov::opset10::Constant::create(ov::element::f32, {}, {-1});
        auto div = std::make_shared<ov::opset10::Power>(sqrt, div_const);
        auto mul1 = std::make_shared<ov::opset10::Multiply>(input, div);
        auto gamma = ov::opset10::Constant::create(ov::element::f32, ov::Shape{6}, {0.029f, 0.014f, 0.003f, 0.013f, 0.015f, 0.009f});
        auto mul2 = std::make_shared<ov::opset10::Multiply>(gamma, mul1);
        auto comp = std::make_shared<ov::opset10::Convert>(mul2, ov::element::f16);

        model = std::make_shared<ov::Model>(ov::NodeVector{comp}, ov::ParameterVector{input});
        manager.register_pass<RMSFusion>();
    }
    {
        auto input = std::make_shared<ov::opset10::Parameter>(ov::element::f32, ov::Shape{1, 2, 6});
        auto rms_const = ov::opset10::Constant::create(ov::element::f32, ov::Shape{6}, {0.029f, 0.014f, 0.003f, 0.013f, 0.015f, 0.009f});
        auto rms = std::make_shared<op::RMS>(input, rms_const, 1e-5f, ov::element::f16);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{rms}, ov::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, RMSNormFusionTest2) {
    {
        auto input = std::make_shared<ov::opset10::Parameter>(ov::element::f32, ov::Shape{1, 2, 6});
        auto power_const = ov::opset10::Constant::create(ov::element::f32, {}, {2.f});
        auto power = std::make_shared<ov::opset10::Power>(input, power_const);
        auto mean_axes = ov::opset10::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
        auto mean = std::make_shared<ov::opset10::ReduceMean>(power, mean_axes, true);
        auto eps = ov::opset10::Constant::create(ov::element::f32, {}, {1e-5f});
        auto add_eps = std::make_shared<ov::opset10::Add>(mean, eps);
        auto sqrt = std::make_shared<ov::opset10::Sqrt>(add_eps);
        auto div_const = ov::opset10::Constant::create(ov::element::f32, {}, {1});
        auto div = std::make_shared<ov::opset10::Divide>(div_const, sqrt);
        auto mul1 = std::make_shared<ov::opset10::Multiply>(input, div);
        auto gamma = ov::opset10::Constant::create(ov::element::f32, ov::Shape{6}, {0.029f, 0.014f, 0.003f, 0.013f, 0.015f, 0.009f});
        auto mul2 = std::make_shared<ov::opset10::Multiply>(gamma, mul1);
        auto comp = std::make_shared<ov::opset10::Convert>(mul2, ov::element::f16);

        model = std::make_shared<ov::Model>(ov::NodeVector{comp}, ov::ParameterVector{input});
        manager.register_pass<RMSFusion>();
    }
}

TEST_F(TransformationTestsF, RMSNormFusionTest3) {
    {
        auto input = std::make_shared<ov::opset10::Parameter>(ov::element::f32, ov::Shape{1, 2, 6});
        auto power_const = ov::opset10::Constant::create(ov::element::f32, {}, {2.f});
        auto power = std::make_shared<ov::opset10::Power>(input, power_const);
        auto mean_axes = ov::opset10::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
        auto mean = std::make_shared<ov::opset10::ReduceMean>(power, mean_axes, true);
        auto eps = ov::opset10::Constant::create(ov::element::f32, {}, {1e-5f});
        auto add_eps = std::make_shared<ov::opset10::Add>(mean, eps);
        auto sqrt = std::make_shared<ov::opset10::Sqrt>(add_eps);
        auto div_const = ov::opset10::Constant::create(ov::element::f32, {}, {1});
        auto div = std::make_shared<ov::opset10::Power>(sqrt, div_const);
        auto mul1 = std::make_shared<ov::opset10::Multiply>(input, div);
        auto gamma = ov::opset10::Constant::create(ov::element::f32, ov::Shape{6}, {0.029f, 0.014f, 0.003f, 0.013f, 0.015f, 0.009f});
        auto mul2 = std::make_shared<ov::opset10::Multiply>(gamma, mul1);
        auto comp = std::make_shared<ov::opset10::Convert>(mul2, ov::element::f16);

        model = std::make_shared<ov::Model>(ov::NodeVector{comp}, ov::ParameterVector{input});
        manager.register_pass<RMSFusion>();
    }
}

TEST_F(TransformationTestsF, RMSNormFusionTest4) {
    {
        auto input = std::make_shared<ov::opset10::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, 6});
        auto power_const = ov::opset10::Constant::create(ov::element::f32, {}, {2.f});
        auto power = std::make_shared<ov::opset10::Power>(input, power_const);
        auto mean_axes = ov::opset10::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
        auto mean = std::make_shared<ov::opset10::ReduceMean>(power, mean_axes, true);
        auto eps = ov::opset10::Constant::create(ov::element::f32, {}, {1e-5f});
        auto add_eps = std::make_shared<ov::opset10::Add>(mean, eps);
        auto sqrt = std::make_shared<ov::opset10::Sqrt>(add_eps);
        auto div_const = ov::opset10::Constant::create(ov::element::f32, {}, {1});
        auto div = std::make_shared<ov::opset10::Divide>(div_const, sqrt);
        auto mul1 = std::make_shared<ov::opset10::Multiply>(input, div);
        auto gamma = ov::opset10::Constant::create(ov::element::f32, ov::Shape{6}, {0.029f, 0.014f, 0.003f, 0.013f, 0.015f, 0.009f});
        auto mul2 = std::make_shared<ov::opset10::Multiply>(gamma, mul1);
        auto comp = std::make_shared<ov::opset10::Convert>(mul2, ov::element::f16);

        model = std::make_shared<ov::Model>(ov::NodeVector{comp}, ov::ParameterVector{input});
        manager.register_pass<RMSFusion>();
    }
}

TEST_F(TransformationTestsF, RMSNormFusionTest5) {
    {
        auto input = std::make_shared<ov::opset10::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, 6});
        auto power_const = ov::opset10::Constant::create(ov::element::f32, {}, {2.f});
        auto power = std::make_shared<ov::opset10::Power>(input, power_const);
        auto mean_axes = ov::opset10::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
        auto mean = std::make_shared<ov::opset10::ReduceMean>(power, mean_axes, true);
        auto eps = ov::opset10::Constant::create(ov::element::f32, {}, {1e-5f});
        auto add_eps = std::make_shared<ov::opset10::Add>(mean, eps);
        auto sqrt = std::make_shared<ov::opset10::Sqrt>(add_eps);
        auto div_const = ov::opset10::Constant::create(ov::element::f32, {}, {-1});
        auto div = std::make_shared<ov::opset10::Power>(sqrt, div_const);
        auto mul1 = std::make_shared<ov::opset10::Multiply>(input, div);
        auto gamma = ov::opset10::Constant::create(ov::element::f32, ov::Shape{6}, {0.029f, 0.014f, 0.003f, 0.013f, 0.015f, 0.009f});
        auto mul2 = std::make_shared<ov::opset10::Multiply>(gamma, mul1);
        auto comp = std::make_shared<ov::opset10::Convert>(mul2, ov::element::f16);

        model = std::make_shared<ov::Model>(ov::NodeVector{comp}, ov::ParameterVector{input});
        manager.register_pass<RMSFusion>();
    }
    {
        auto input = std::make_shared<ov::opset10::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, 6});
        auto rms_const = ov::opset10::Constant::create(ov::element::f32, ov::Shape{6}, {0.029f, 0.014f, 0.003f, 0.013f, 0.015f, 0.009f});
        auto rms = std::make_shared<op::RMS>(input, rms_const, 1e-5f, ov::element::f16);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{rms}, ov::ParameterVector{input});
    }
}
