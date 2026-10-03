// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/transformations/keep_precision_sensitive_bf16.hpp"

#include <gtest/gtest.h>

#include <memory>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/manager.hpp"
#include "ov_ops/rms.hpp"
#include "transformations/convert_precision.hpp"
#include "transformations/rt_info/disable_precision_conversion.hpp"

using namespace testing;
using namespace ov::intel_gpu;

namespace {

constexpr double rms_eps = 1e-6;
constexpr size_t head_size = 64;

std::shared_ptr<ov::op::internal::RMS> make_rms(const ov::Output<ov::Node>& data) {
    auto gamma = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{head_size}, {1.0f});
    return std::make_shared<ov::op::internal::RMS>(data, gamma, rms_eps, ov::element::f32);
}

std::shared_ptr<ov::op::v1::Transpose> make_transpose(const ov::Output<ov::Node>& data, const std::vector<int32_t>& order) {
    auto order_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{order.size()}, order);
    return std::make_shared<ov::op::v1::Transpose>(data, order_const);
}

std::shared_ptr<ov::op::v1::Reshape> make_reshape(const ov::Output<ov::Node>& data, const std::vector<int32_t>& shape) {
    auto shape_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{shape.size()}, shape);
    return std::make_shared<ov::op::v1::Reshape>(data, shape_const, false);
}

void run_pass(const std::shared_ptr<ov::Model>& model) {
    ov::pass::Manager manager;
    manager.register_pass<KeepPrecisionSensitiveSubgraphsForBF16>();
    manager.run_passes(model);
}

}  // namespace

TEST(KeepPrecisionSensitiveBF16Test, MarksQKNormOnQueryPath) {
    auto q_in = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 8, 16, head_size});
    auto k_in = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 16, 8, head_size});
    auto v_in = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 16, 8, head_size});

    auto rms = make_rms(q_in);
    auto transpose = make_transpose(rms, {0, 2, 1, 3});
    auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(transpose, k_in, v_in, false);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{sdpa}, ov::ParameterVector{q_in, k_in, v_in});

    run_pass(model);

    EXPECT_TRUE(ov::is_conversion_disabled(rms, ov::element::bf16)) << "the query normalization was not marked to be kept in f32";
    // The attention itself is not part of the island: only the normalization is.
    EXPECT_FALSE(ov::is_conversion_disabled(sdpa, ov::element::bf16));
}

TEST(KeepPrecisionSensitiveBF16Test, MarksQKNormOnKeyPathThroughGlueChain) {
    auto q_in = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 8, 16, head_size});
    auto k_in = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 4, 8, head_size});
    auto v_in = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 8, 16, head_size});

    auto rms = make_rms(k_in);
    auto unsqueeze_axes = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {2});
    auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(rms, unsqueeze_axes);
    auto broadcast_shape = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{5}, {1, 4, 4, 8, 64});
    auto broadcast = std::make_shared<ov::op::v3::Broadcast>(unsqueeze, broadcast_shape);
    auto reshape = make_reshape(broadcast, {1, 16, 8, 64});
    auto transpose = make_transpose(reshape, {0, 2, 1, 3});
    auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(q_in, transpose, v_in, false);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{sdpa}, ov::ParameterVector{q_in, k_in, v_in});

    run_pass(model);

    EXPECT_TRUE(ov::is_conversion_disabled(rms, ov::element::bf16)) << "the key normalization was not marked to be kept in f32";
}

TEST(KeepPrecisionSensitiveBF16Test, DoesNotMarkNormBehindMatMul) {
    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 16, head_size});
    auto k_in = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 8, 16, head_size});
    auto v_in = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 8, 16, head_size});

    auto rms = make_rms(data);
    auto weights = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{head_size, 512}, {0.1f});
    auto matmul = std::make_shared<ov::op::v0::MatMul>(rms, weights);
    auto reshape = make_reshape(matmul, {1, 16, 8, 64});
    auto transpose = make_transpose(reshape, {0, 2, 1, 3});
    auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(transpose, k_in, v_in, false);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{sdpa}, ov::ParameterVector{data, k_in, v_in});

    run_pass(model);

    EXPECT_FALSE(ov::is_conversion_disabled(rms, ov::element::bf16)) << "a normalization feeding the query projection must not be kept in f32";
}

TEST(KeepPrecisionSensitiveBF16Test, DoesNotMarkNormOnValuePath) {
    auto q_in = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 16, 8, head_size});
    auto k_in = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 16, 8, head_size});
    auto v_in = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 8, 16, head_size});

    auto rms = make_rms(v_in);
    auto transpose = make_transpose(rms, {0, 2, 1, 3});
    auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(q_in, k_in, transpose, false);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{sdpa}, ov::ParameterVector{q_in, k_in, v_in});

    run_pass(model);

    EXPECT_FALSE(ov::is_conversion_disabled(rms, ov::element::bf16)) << "a normalization feeding the attention value input must not be kept in f32";
}

TEST(KeepPrecisionSensitiveBF16Test, MirrorsExistingF16MarksToBF16) {
    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 3, 16, 16});
    auto scale = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {0.5f});
    auto marked_mul = std::make_shared<ov::op::v1::Multiply>(data, scale);
    auto other_scale = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {2.0f});
    auto plain_mul = std::make_shared<ov::op::v1::Multiply>(marked_mul, other_scale);
    auto res = std::make_shared<ov::op::v0::Result>(plain_mul);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{data});

    ov::disable_conversion(marked_mul, ov::element::f16);
    ASSERT_FALSE(ov::is_conversion_disabled(marked_mul, ov::element::bf16));

    run_pass(model);

    EXPECT_TRUE(ov::is_conversion_disabled(marked_mul, ov::element::bf16)) << "an existing f16 keep-mark was not mirrored onto the bf16 target";
    EXPECT_TRUE(ov::is_conversion_disabled(marked_mul, ov::element::f16)) << "the original f16 keep-mark was dropped";
    EXPECT_FALSE(ov::is_conversion_disabled(plain_mul, ov::element::f16)) << "an unmarked node got an f16 keep-mark";
}

TEST(KeepPrecisionSensitiveBF16Test, ConvertPrecisionInsertsExactlyOneBoundaryConvertPerTransition) {
    auto q_in = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 8, 16, head_size});
    auto k_in = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 16, 8, head_size});
    auto v_in = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 16, 8, head_size});

    auto rms = make_rms(q_in);
    auto transpose = make_transpose(rms, {0, 2, 1, 3});
    auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(transpose, k_in, v_in, false);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{sdpa}, ov::ParameterVector{q_in, k_in, v_in});

    ov::pass::Manager manager;
    manager.register_pass<KeepPrecisionSensitiveSubgraphsForBF16>();
    manager.register_pass<ov::pass::ConvertPrecision>(ov::element::Type_t::f32,
                                                      ov::element::Type_t::bf16,
                                                      type_to_fuse_map{},
                                                      /*keep_precision_sensitive_in_fp32=*/true,
                                                      /*convert_input_output_precision=*/false);
    manager.run_passes(model);

    auto rms_consumers = rms->output(0).get_target_inputs();
    ASSERT_EQ(rms_consumers.size(), 1u);
    auto* boundary_node = rms_consumers.begin()->get_node();
    ASSERT_TRUE(ov::is_type<ov::op::v0::Convert>(boundary_node))
        << "expected a single boundary Convert right after the kept-f32 RMS, found " << boundary_node->get_type_name();

    auto boundary_consumers = boundary_node->output(0).get_target_inputs();
    ASSERT_EQ(boundary_consumers.size(), 1u);
    EXPECT_TRUE(ov::is_type<ov::op::v1::Transpose>(boundary_consumers.begin()->get_node()))
        << "a duplicate boundary Convert was inserted between the kept RMS and its consumer";
}
