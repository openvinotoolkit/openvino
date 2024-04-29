// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <limits>
#include <string>
#include <memory>

#include <openvino/pass/manager.hpp>
#include <openvino/core/model.hpp>
#include "openvino/core/coordinate_diff.hpp"
#include "openvino/core/type/element_type.hpp"
#include <openvino/op/constant.hpp>
#include "openvino/op/clamp.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/softmax.hpp"
#include <plugin/transformations/clamp_fp16_output.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ov_test_utils.hpp"

using namespace testing;
using namespace ov::intel_gpu;

TEST_F(TransformationTestsF, ClampFp16OutputTest1) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{ 3, 2, 2 });
        auto input2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{ 1, 2, 2 });
        auto matmul = std::make_shared<ov::op::v0::MatMul>(input1, input2, true, false);
        auto softmax = std::make_shared<ov::op::v8::Softmax>(matmul, 1);

        model = std::make_shared<ov::Model>(ov::NodeVector{ softmax }, ov::ParameterVector{ input1, input2 });
        manager.register_pass<ClampFP16Output>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{ 3, 2, 2 });
        auto input2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{ 1, 2, 2 });
        auto matmul = std::make_shared<ov::op::v0::MatMul>(input1, input2, true, false);
        auto min = static_cast<double>(std::numeric_limits<ov::float16>::lowest());
        auto max = static_cast<double>(std::numeric_limits<ov::float16>::max());
        auto clamp = std::make_shared<ov::op::v0::Clamp>(matmul, min, max);
        auto softmax = std::make_shared<ov::op::v8::Softmax>(clamp, 1);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{ softmax }, ov::ParameterVector{ input1, input2 });
    }
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, ClampFp16OutputTest2) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{ 3, 2, 2 });
        auto input2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{ 1, 2, 2 });
        auto matmul = std::make_shared<ov::op::v0::MatMul>(input1, input2, true, false);
        auto target_shape = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 2 }, { 3, 4 });
        auto reshape = std::make_shared<ov::op::v1::Reshape>(matmul, target_shape, false);
        auto softmax = std::make_shared<ov::op::v8::Softmax>(reshape, 1);

        model = std::make_shared<ov::Model>(ov::NodeVector{ softmax }, ov::ParameterVector{ input1, input2 });
        manager.register_pass<ClampFP16Output>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{ 3, 2, 2 });
        auto input2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{ 1, 2, 2 });
        auto matmul = std::make_shared<ov::op::v0::MatMul>(input1, input2, true, false);
        auto min = static_cast<double>(std::numeric_limits<ov::float16>::lowest());
        auto max = static_cast<double>(std::numeric_limits<ov::float16>::max());
        auto clamp = std::make_shared<ov::op::v0::Clamp>(matmul, min, max);
        auto target_shape = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 2 }, { 3, 4 });
        auto reshape = std::make_shared<ov::op::v1::Reshape>(clamp, target_shape, false);
        auto softmax = std::make_shared<ov::op::v8::Softmax>(reshape, 1);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{ softmax }, ov::ParameterVector{ input1, input2 });
    }
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, ClampFp16OutputTest3) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{ 3, 2, 2 });
        auto input2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{ 1, 2, 2 });
        auto matmul = std::make_shared<ov::op::v0::MatMul>(input1, input2, true, false);
        auto softmax = std::make_shared<ov::op::v8::Softmax>(matmul, 1);

        model = std::make_shared<ov::Model>(ov::NodeVector{ softmax }, ov::ParameterVector{ input1, input2 });
        manager.register_pass<ClampFP16Output>();
    }
    {
        model_ref = model->clone(); // not changed due to f32 precision
    }
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, ClampFp16OutputTest4) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{ 3, 2, 2 });
        auto input2 = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 1, 2, 2 }, { 1 });
        auto matmul = std::make_shared<ov::op::v0::MatMul>(input1, input2, true, false);
        auto softmax = std::make_shared<ov::op::v8::Softmax>(matmul, 1);

        model = std::make_shared<ov::Model>(ov::NodeVector{ softmax }, ov::ParameterVector{ input1 });
        manager.register_pass<ClampFP16Output>();
    }
    {
        model_ref = model->clone(); // Not changed due to const input2
    }
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, ClampFp16OutputTest5) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{ 3, 2, 2 });
        auto input2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{ 1, 2, 2 });
        auto matmul = std::make_shared<ov::op::v0::MatMul>(input1, input2, true, false);
        auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{ 3, 2, 2 });
        auto add = std::make_shared<ov::op::v1::Add>(matmul, data);
        auto softmax = std::make_shared<ov::op::v8::Softmax>(add, 1);

        model = std::make_shared<ov::Model>(ov::NodeVector{ softmax }, ov::ParameterVector{ input1, input2, data });
        manager.register_pass<ClampFP16Output>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{ 3, 2, 2 });
        auto input2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{ 1, 2, 2 });
        auto matmul = std::make_shared<ov::op::v0::MatMul>(input1, input2, true, false);
        auto min = static_cast<double>(std::numeric_limits<ov::float16>::lowest());
        auto max = static_cast<double>(std::numeric_limits<ov::float16>::max());
        auto clamp = std::make_shared<ov::op::v0::Clamp>(matmul, min, max);
        auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{ 3, 2, 2 });
        auto add = std::make_shared<ov::op::v1::Add>(clamp, data);
        auto softmax = std::make_shared<ov::op::v8::Softmax>(add, 1);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{ softmax }, ov::ParameterVector{ input1, input2, data });
    }
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, ClampFp16OutputTest6) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{ 3, 2, 2 });
        auto input2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{ 1, 2, 2 });
        auto matmul = std::make_shared<ov::op::v0::MatMul>(input1, input2, true, false);
        auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{ 3, 2, 2 });
        auto maximum = std::make_shared<ov::op::v1::Maximum>(matmul, data);
        auto softmax = std::make_shared<ov::op::v8::Softmax>(maximum, 1);

        model = std::make_shared<ov::Model>(ov::NodeVector{ softmax }, ov::ParameterVector{ input1, input2, data });
        manager.register_pass<ClampFP16Output>();
    }
    {
        model_ref = model->clone(); // Not changed due to types for eltwise not supporting fusion to gemm
    }
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}
