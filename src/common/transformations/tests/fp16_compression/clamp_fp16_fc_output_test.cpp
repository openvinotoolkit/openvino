// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/fp16_compression/clamp_fp16_fc_output.hpp"

#include <gtest/gtest.h>

#include <limits>
#include <memory>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"

using namespace testing;
using namespace ov::pass;

namespace {
double clamp_min() {
    return static_cast<double>(std::numeric_limits<ov::float16>::lowest());
}
double clamp_max() {
    return static_cast<double>(std::numeric_limits<ov::float16>::max());
}
}  // namespace

TEST_F(TransformationTestsF, ClampFp16FCOutputTest1_ConstantWeightMatMulIntoAdd) {
    {
        auto activation = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 128, 4096});
        auto weight = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{4096, 4096}, {1});
        auto matmul = std::make_shared<ov::op::v0::MatMul>(activation, weight, false, true);
        auto residual = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 128, 4096});
        auto add = std::make_shared<ov::op::v1::Add>(matmul, residual);

        model = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{activation, residual});
        manager.register_pass<ClampFP16FCOutput>();
    }
    {
        auto activation = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 128, 4096});
        auto weight = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{4096, 4096}, {1});
        auto matmul = std::make_shared<ov::op::v0::MatMul>(activation, weight, false, true);
        auto clamp = std::make_shared<ov::op::v0::Clamp>(matmul, clamp_min(), clamp_max());
        auto residual = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 128, 4096});
        auto add = std::make_shared<ov::op::v1::Add>(clamp, residual);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{activation, residual});
    }
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, ClampFp16FCOutputTest2_ConstantWeightMatMulConvertIntoAdd) {
    {
        auto activation = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 128, 4096});
        auto weight = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{4096, 4096}, {1});
        auto matmul = std::make_shared<ov::op::v0::MatMul>(activation, weight, false, true);
        auto convert = std::make_shared<ov::op::v0::Convert>(matmul, ov::element::f32);
        auto residual = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 128, 4096});
        auto add = std::make_shared<ov::op::v1::Add>(convert, residual);

        model = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{activation, residual});
        manager.register_pass<ClampFP16FCOutput>();
    }
    {
        auto activation = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 128, 4096});
        auto weight = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{4096, 4096}, {1});
        auto matmul = std::make_shared<ov::op::v0::MatMul>(activation, weight, false, true);
        auto convert = std::make_shared<ov::op::v0::Convert>(matmul, ov::element::f32);
        auto clamp = std::make_shared<ov::op::v0::Clamp>(convert, clamp_min(), clamp_max());
        auto residual = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 128, 4096});
        auto add = std::make_shared<ov::op::v1::Add>(clamp, residual);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{activation, residual});
    }
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, ClampFp16FCOutputTest3_F32NotChanged) {
    {
        auto activation = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 128, 4096});
        auto weight = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{4096, 4096}, {1});
        auto matmul = std::make_shared<ov::op::v0::MatMul>(activation, weight, false, true);
        auto residual = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 128, 4096});
        auto add = std::make_shared<ov::op::v1::Add>(matmul, residual);

        model = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{activation, residual});
        manager.register_pass<ClampFP16FCOutput>();
    }
    { model_ref = model->clone(); }  // not changed due to f32 precision
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, ClampFp16FCOutputTest4_MultiConsumerMatMulNotChanged) {
    {
        auto activation = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 128, 4096});
        auto weight = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{4096, 4096}, {1});
        auto matmul = std::make_shared<ov::op::v0::MatMul>(activation, weight, false, true);
        auto residual = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 128, 4096});
        auto add = std::make_shared<ov::op::v1::Add>(matmul, residual);
        auto other_consumer = std::make_shared<ov::op::v1::Add>(matmul, residual);

        model = std::make_shared<ov::Model>(ov::OutputVector{add, other_consumer},
                                             ov::ParameterVector{activation, residual});
        manager.register_pass<ClampFP16FCOutput>();
    }
    { model_ref = model->clone(); }
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, ClampFp16FCOutputTest5_NonConstantWeightNotChanged) {
    {
        auto activation = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 128, 4096});
        auto weight = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{4096, 4096});
        auto matmul = std::make_shared<ov::op::v0::MatMul>(activation, weight, false, true);
        auto residual = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 128, 4096});
        auto add = std::make_shared<ov::op::v1::Add>(matmul, residual);

        model =
            std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{activation, weight, residual});
        manager.register_pass<ClampFP16FCOutput>();
    }
    { model_ref = model->clone(); }  // not changed: weight operand must be a Constant
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}
