// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset1_decl.hpp"
#include "openvino/pass/manager.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "transformations/cpu_opset/common/pass/convert_matmul_to_fc.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/constant.hpp"
#include "ov_ops/fully_connected.hpp"
#include "openvino/op/matmul.hpp"

using namespace testing;
using namespace ov::intel_cpu;

TEST_F(TransformationTestsF, ConvertMatMulToFCTest5) {
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, 2});
        auto input2 = ov::opset1::Constant::create(ov::element::f32, ov::Shape{3, 2, 2}, {1});
        auto matmul = std::make_shared<ov::opset1::MatMul>(input1, input2, false, true);

        model = std::make_shared<ov::Model>(ov::OutputVector{matmul}, ov::ParameterVector{input1});
        manager.register_pass<ConvertMatMulToFC>();
    }
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, 2});
        auto input2 = ov::opset1::Constant::create(ov::element::f32, ov::Shape{3, 2, 2}, {1});
        auto matmul = std::make_shared<ov::op::internal::FullyConnected>(
            input1,
            input2,
            std::make_shared<ov::op::v0::Constant>(ov::element::dynamic, ov::Shape{0}));

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{matmul}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertMatMulToFCTest6) {
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, 2});
        auto input2 = ov::opset1::Constant::create(ov::element::f32, ov::Shape{3, 1, 2}, {1});
        auto matmul = std::make_shared<ov::opset1::MatMul>(input1, input2, false, true);

        model = std::make_shared<ov::Model>(ov::OutputVector{matmul}, ov::ParameterVector{input1});
        manager.register_pass<ConvertMatMulToFC>();
    }
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, 2});
        auto input2 = ov::opset1::Constant::create(ov::element::f32, ov::Shape{3, 1, 2}, {1});
        auto matmul = std::make_shared<ov::op::internal::FullyConnected>(
            input1,
            input2,
            std::make_shared<ov::op::v0::Constant>(ov::element::dynamic, ov::Shape{0}));

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{matmul}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertMatMulToFCTest11) {
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{18, -1, 1});
        auto input2 = ov::opset1::Constant::create(ov::element::f32, ov::Shape{18, 80, 1}, {1});
        auto matmul = std::make_shared<ov::opset1::MatMul>(input1, input2, false, true);

        model = std::make_shared<ov::Model>(ov::OutputVector{matmul}, ov::ParameterVector{input1});
        manager.register_pass<ConvertMatMulToFC>();
    }
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{18, -1, 1});
        auto input2 = ov::opset1::Constant::create(ov::element::f32, ov::Shape{18, 80, 1}, {1});
        auto matmul = std::make_shared<ov::op::internal::FullyConnected>(
            input1,
            input2,
            std::make_shared<ov::op::v0::Constant>(ov::element::dynamic, ov::Shape{0}));

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{matmul}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, ConvertMatMulToFCTest12) {
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{1, -1, 1});
        auto input2 = ov::opset1::Constant::create(ov::element::f32, ov::Shape{2, 80, 1}, {1});
        auto matmul = std::make_shared<ov::opset1::MatMul>(input1, input2, false, true);

        model = std::make_shared<ov::Model>(ov::OutputVector{matmul}, ov::ParameterVector{input1});
        manager.register_pass<ConvertMatMulToFC>();
    }
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{1, -1, 1});
        auto input2 = ov::opset1::Constant::create(ov::element::f32, ov::Shape{2, 80, 1}, {1});
        auto matmul = std::make_shared<ov::op::internal::FullyConnected>(
            input1,
            input2,
            std::make_shared<ov::op::v0::Constant>(ov::element::dynamic, ov::Shape{0}));

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{matmul}, ov::ParameterVector{input1});
    }
}