// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <openvino/core/model.hpp>
#include <openvino/pass/manager.hpp>
#include "common_test_utils/ov_test_utils.hpp"
#include <transformations/utils/utils.hpp>

#include <plugin/transformations/fc_convert_fusion.hpp>

#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/parameter.hpp"
#include "intel_gpu/op/fully_connected.hpp"
#include "intel_gpu/op/fully_connected_compressed.hpp"
#include "intel_gpu/op/placeholder.hpp"
#include "openvino/op/variadic_split.hpp"

using namespace testing;
using namespace ov::intel_gpu;

TEST_F(TransformationTestsF, FullyConnectedConvertFusionTest1) {
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 32, 16 }, { 1 });
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 32, 1 }, { 1 });
        auto zp_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 32, 1 }, { 1 });
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input, weights_const, no_bias, scale_const, zp_const);
        auto convert = std::make_shared<ov::op::v0::Convert>(fc_compressed, ov::element::f32);

        model = std::make_shared<ov::Model>(ov::OutputVector{convert}, ov::ParameterVector{input});
        manager.register_pass<FullyConnectedConvertFusion>();
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 32, 16 }, { 1 });
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 32, 1 }, { 1 });
        auto zp_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 32, 1 }, { 1 });
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input, weights_const, no_bias, scale_const, zp_const, ov::element::f32);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, FullyConnectedConvertFusionTest2) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{3, 2, 2});
        auto input2 = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{2, 2}, {1});
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto matmul = std::make_shared<op::FullyConnected>(input1, input2, no_bias);
        auto convert = std::make_shared<ov::op::v0::Convert>(matmul, ov::element::f32);

        model = std::make_shared<ov::Model>(ov::OutputVector{convert}, ov::ParameterVector{input1});
        manager.register_pass<FullyConnectedConvertFusion>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{3, 2, 2});
        auto input2 = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{2, 2}, {1});
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto matmul = std::make_shared<op::FullyConnected>(input1, input2, no_bias, ov::element::f32);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{matmul}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, FullyConnectedConvertFusionTest3) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 163, 960});
        auto axis_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
        auto split_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {113, 50});
        auto split = std::make_shared<ov::op::v1::VariadicSplit>(input1, axis_const, split_const);
        auto input2 = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{720, 960}, {1});
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();

        bool default_output_mismatch_exception = false;
        try {
            auto matmul = std::make_shared<op::FullyConnected>(split, input2, no_bias);
        } catch(std::exception& exc) {
            const std::string error = exc.what();
            default_output_mismatch_exception = error.find("Default output not supported") != std::string::npos;
        }

        ASSERT_TRUE(default_output_mismatch_exception);

        auto non_split_input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 50, 960});
        auto matmul = std::make_shared<op::FullyConnected>(non_split_input, input2, no_bias);
        auto convert = std::make_shared<ov::op::v0::Convert>(matmul, ov::element::f32);
        model = std::make_shared<ov::Model>(ov::OutputVector{convert}, ov::ParameterVector{non_split_input});
        manager.register_pass<FullyConnectedConvertFusion>();

    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 50, 960});
        auto input2 = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{720, 960}, {1});
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto matmul = std::make_shared<op::FullyConnected>(input1, input2, no_bias, ov::element::f32);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{matmul}, ov::ParameterVector{input1});
    }
}
