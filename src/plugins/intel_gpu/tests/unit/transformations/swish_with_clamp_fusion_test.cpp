// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "plugin/transformations/swiglu_fusion_with_clamp.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/clamp.hpp"
#include "intel_gpu/op/swiglu_with_clamp.hpp"
#include "openvino/opsets/opset4_decl.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"
using namespace ov;
using namespace testing;
using namespace ov::intel_gpu;
TEST_F(TransformationTestsF, SwishFusionWithClamp) {
    {
        auto reshape = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{32, -1, 2880});
        auto convert1 = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{32, 2880, 5760});
        auto gemm_up = std::make_shared<ov::op::v0::MatMul>(reshape, convert1);
        auto add_const1 = std::make_shared<ov::op::v0::Constant>(element::f32, Shape{32, 1, 5760});
        auto add_gemm = std::make_shared<ov::op::v1::Add>(gemm_up, add_const1);

        auto start1 = ov::op::v0::Constant::create(element::i64, Shape{1}, {0});
        auto stop1 = ov::op::v0::Constant::create(element::i64, Shape{1}, {2880});
        auto step = ov::op::v0::Constant::create(element::i64, Shape{1}, {1});
        auto axis = ov::op::v0::Constant::create(element::i64, Shape{1}, {2});
        auto slice1 = std::make_shared<ov::op::v8::Slice>(add_gemm, start1, stop1, step, axis);

        auto start2 = ov::op::v0::Constant::create(element::i64, Shape{1}, {2880});
        auto stop2 = ov::op::v0::Constant::create(element::i64, Shape{1}, {5760});
        auto slice2 = std::make_shared<ov::op::v8::Slice>(add_gemm, start2, stop2, step, axis);

        auto min_val = std::make_shared<ov::op::v0::Constant>(element::f32, Shape{1, 1, 1});
        auto min = std::make_shared<ov::op::v1::Minimum>(slice1, min_val);
        auto beta = ov::op::v0::Constant::create(element::f32, Shape{}, {1.7f});
        auto swish = std::make_shared<ov::op::v4::Swish>(min, beta);

        auto clamp = std::make_shared<ov::op::v0::Clamp>(slice2, -7.0, 7.0);
        auto add_const2 = std::make_shared<ov::op::v0::Constant>(element::f32, Shape{1, 1, 1});
        auto add_clamp = std::make_shared<ov::op::v1::Add>(clamp, add_const2);

        auto multiply = std::make_shared<ov::op::v1::Multiply>(swish, add_clamp);

        auto convert2 = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{32, 2880, 2880});

        auto gemm_down = std::make_shared<ov::op::v0::MatMul>(multiply, convert2);

        auto result = std::make_shared<ov::op::v0::Result>(gemm_down);

        model = std::make_shared<ov::Model>(ResultVector{result}, ParameterVector{reshape, convert1, convert2});

        manager.register_pass<SwiGluFusionWithClamp>();
    }
    {
        auto reshape = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{32, -1, 2880});
        auto convert1 = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{32, 2880, 5760});
        auto gemm_up = std::make_shared<ov::op::v0::MatMul>(reshape, convert1);
        auto swiglu_with_clamp = std::make_shared<
            ov::intel_gpu::op::SwiGluWithClamp>(gemm_up, 2, 2880, ov::op::internal::GLU::GluType::Swish, 0, -7, 7, 1.7f, 0.0f, ov::element::Type_t::f32);
        auto result = std::make_shared<ov::op::v0::Result>(swiglu_with_clamp);

        model_ref = std::make_shared<ov::Model>(ResultVector{result}, ParameterVector{reshape, convert1});
    }
}

TEST_F(TransformationTestsF, SwishFusionWithClampStrided) {
    {
        auto reshape = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{32, -1, 2880});
        auto convert1 = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{32, 2880, 5760});
        auto gemm_up = std::make_shared<ov::op::v0::MatMul>(reshape, convert1);
        auto add_const1 = std::make_shared<ov::op::v0::Constant>(element::f32, Shape{32, 1, 5760});
        auto add_gemm = std::make_shared<ov::op::v1::Add>(gemm_up, add_const1);

        auto start1 = ov::op::v0::Constant::create(element::i64, Shape{1}, {0});
        auto stop1 = ov::op::v0::Constant::create(element::i64, Shape{1}, {5760});
        auto step1 = ov::op::v0::Constant::create(element::i64, Shape{1}, {2});
        auto axis1 = ov::op::v0::Constant::create(element::i64, Shape{1}, {2});
        auto slice1 = std::make_shared<ov::op::v8::Slice>(add_gemm, start1, stop1, step1, axis1);

        auto start2 = ov::op::v0::Constant::create(element::i64, Shape{1}, {1});
        auto stop2 = ov::op::v0::Constant::create(element::i64, Shape{1}, {5760});
        auto step2 = ov::op::v0::Constant::create(element::i64, Shape{1}, {2});
        auto axis2 = ov::op::v0::Constant::create(element::i64, Shape{1}, {2});
        auto slice2 = std::make_shared<ov::op::v8::Slice>(add_gemm, start2, stop2, step2, axis2);

        auto min_val = std::make_shared<ov::op::v0::Constant>(element::f32, Shape{1, 1, 1});
        auto min = std::make_shared<ov::op::v1::Minimum>(slice1, min_val);
        auto beta = ov::op::v0::Constant::create(element::f32, Shape{}, {1.7f});
        auto swish = std::make_shared<ov::op::v4::Swish>(min, beta);

        auto clamp = std::make_shared<ov::op::v0::Clamp>(slice2, -7.0, 7.0);
        auto add_const2 = std::make_shared<ov::op::v0::Constant>(element::f32, Shape{1, 1, 1});
        auto add_clamp = std::make_shared<ov::op::v1::Add>(clamp, add_const2);

        auto multiply = std::make_shared<ov::op::v1::Multiply>(swish, add_clamp);

        auto convert2 = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{32, 2880, 2880});

        auto gemm_down = std::make_shared<ov::op::v0::MatMul>(multiply, convert2);

        auto result = std::make_shared<ov::op::v0::Result>(gemm_down);

        model = std::make_shared<ov::Model>(ResultVector{result}, ParameterVector{reshape, convert1, convert2});

        manager.register_pass<SwiGluFusionWithClamp>();
    }
    {
        auto reshape = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{32, -1, 2880});
        auto convert1 = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{32, 2880, 5760});
        auto gemm_up = std::make_shared<ov::op::v0::MatMul>(reshape, convert1);
        auto swiglu_with_clamp = std::make_shared<
            ov::intel_gpu::op::SwiGluWithClamp>(gemm_up, 2, 2, ov::op::internal::GLU::GluType::Swish, 0, -7, 7, 1.7f, 0.0f, ov::element::Type_t::f32);
        auto result = std::make_shared<ov::op::v0::Result>(swiglu_with_clamp);

        model_ref = std::make_shared<ov::Model>(ResultVector{result}, ParameterVector{reshape, convert1});
    }
}
