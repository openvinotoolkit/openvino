// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/moe.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/swish.hpp"
#include "ov_ops/moe_compressed.hpp"
#include "plugin/transformations/fuse_moe_shared_expert.hpp"

using namespace testing;
using namespace ov::intel_gpu;

namespace ov {
namespace test {
namespace intel_gpu {

// Helper to create a weight decompression chain: Const(u4) -> Convert(f16) -> Sub(zp) -> Mul(scale) -> Reshape -> Convert(f32)
static ov::Output<ov::Node> create_decompression_chain(
    ov::element::Type wei_type,
    const ov::Shape& wei_shape,
    const ov::Shape& zp_shape,
    const ov::Shape& scale_shape,
    const ov::Shape& reshape_target,
    float wei_val = 1.0f,
    float zp_val = 0.0f,
    float scale_val = 0.01f) {
    auto wei = op::v0::Constant::create(wei_type, wei_shape, {wei_val});
    auto zp = op::v0::Constant::create(wei_type, zp_shape, {zp_val});
    auto scale = op::v0::Constant::create(element::f16, scale_shape, {scale_val});
    auto reshape_const = op::v0::Constant::create(element::i32, Shape{reshape_target.size()}, reshape_target);

    auto w_f16 = std::make_shared<op::v0::Convert>(wei, element::f16);
    auto zp_f16 = std::make_shared<op::v0::Convert>(zp, element::f16);
    auto sub = std::make_shared<op::v1::Subtract>(w_f16, zp_f16);
    auto mul = std::make_shared<op::v1::Multiply>(sub, scale);
    auto reshape = std::make_shared<op::v1::Reshape>(mul, reshape_const, false);
    auto convert = std::make_shared<op::v0::Convert>(reshape, element::f32);
    return convert;
}

// Helper to create a symmetric weight decompression chain: Const -> Convert(f16) -> Mul(scale) -> Reshape -> Convert(f32)
static ov::Output<ov::Node> create_sym_decompression_chain(
    ov::element::Type wei_type,
    const ov::Shape& wei_shape,
    const ov::Shape& scale_shape,
    const ov::Shape& reshape_target,
    float wei_val = 1.0f,
    float scale_val = 0.01f) {
    auto wei = op::v0::Constant::create(wei_type, wei_shape, {wei_val});
    auto scale = op::v0::Constant::create(element::f16, scale_shape, {scale_val});
    auto reshape_const = op::v0::Constant::create(element::i32, Shape{reshape_target.size()}, reshape_target);

    auto w_f16 = std::make_shared<op::v0::Convert>(wei, element::f16);
    auto mul = std::make_shared<op::v1::Multiply>(w_f16, scale);
    auto reshape = std::make_shared<op::v1::Reshape>(mul, reshape_const, false);
    auto convert = std::make_shared<op::v0::Convert>(reshape, element::f32);
    return convert;
}

TEST_F(TransformationTestsF, FuseMOESharedExpertWithSigmoidGating) {
    disable_rt_info_check();
    {
        // tokens:32, hidden_size:2048, inter_size:768, experts:128, topk:8
        auto hidden_states = std::make_shared<ov::op::v0::Parameter>(element::f16, Shape{32, 2048});
        auto routing_weights = std::make_shared<ov::op::v0::Parameter>(element::f16, Shape{128, 1, 32, 1});
        auto routing_idx = std::make_shared<ov::op::v0::Parameter>(element::i32, Shape{32, 8});

        // MOE expert decompression chains
        auto gate_decompressed = create_decompression_chain(
            element::u4, {128, 768, 16, 128}, {128, 768, 16, 1}, {128, 768, 16, 1}, {128, 768, 2048});
        auto up_decompressed = create_decompression_chain(
            element::u4, {128, 768, 16, 128}, {128, 768, 16, 1}, {128, 768, 16, 1}, {128, 768, 2048});
        auto down_decompressed = create_decompression_chain(
            element::u4, {128, 2048, 6, 128}, {128, 2048, 6, 1}, {128, 2048, 6, 1}, {128, 2048, 768});

        // MOE node
        ov::op::internal::MOE::Config config;
        config.expert_type = ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU;
        auto hidden_states_f32 = std::make_shared<ov::op::v0::Convert>(hidden_states, element::f32);
        auto moe = std::make_shared<ov::op::internal::MOE>(
            ov::OutputVector{hidden_states_f32, routing_weights, routing_idx,
                             gate_decompressed, up_decompressed, down_decompressed}, config);

        // Shared expert (uses reshaped hidden_states, separate from MOE's input)
        auto reshape_const_hs = op::v0::Constant::create(element::i64, Shape{2}, {32, 2048});
        auto hidden_states_reshaped = std::make_shared<ov::op::v1::Reshape>(hidden_states_f32, reshape_const_hs, false);

        auto sh_gate_decompressed = create_decompression_chain(
            element::u4, {768, 16, 128}, {768, 16, 1}, {768, 16, 1}, {768, 2048}, 2.0f, 1.0f, 0.02f);
        auto sh_up_decompressed = create_decompression_chain(
            element::u4, {768, 16, 128}, {768, 16, 1}, {768, 16, 1}, {768, 2048}, 2.0f, 1.0f, 0.02f);
        auto sh_down_decompressed = create_decompression_chain(
            element::u4, {2048, 6, 128}, {2048, 6, 1}, {2048, 6, 1}, {2048, 768}, 2.0f, 1.0f, 0.02f);

        auto shared_gate_m = std::make_shared<ov::op::v0::MatMul>(hidden_states_reshaped, sh_gate_decompressed, false, true);
        auto shared_swish_m = std::make_shared<ov::op::v4::Swish>(shared_gate_m);
        auto shared_up_m = std::make_shared<ov::op::v0::MatMul>(hidden_states_reshaped, sh_up_decompressed, false, true);
        auto shared_mul_m = std::make_shared<ov::op::v1::Multiply>(shared_swish_m, shared_up_m);
        auto shared_down_m = std::make_shared<ov::op::v0::MatMul>(shared_mul_m, sh_down_decompressed, false, true);

        // Sigmoid gating
        auto gate_gate_wei = std::make_shared<ov::op::v0::Constant>(element::f32, Shape{2048, 1}, std::vector<float>(2048, 1.0f));
        auto gate_gate_mm = std::make_shared<ov::op::v0::MatMul>(hidden_states_reshaped, gate_gate_wei);
        auto gate_sigmoid = std::make_shared<ov::op::v0::Sigmoid>(gate_gate_mm);
        auto shared_gated = std::make_shared<ov::op::v1::Multiply>(gate_sigmoid, shared_down_m);

        // Reshape + Add
        auto reshape_const_output = op::v0::Constant::create(element::i64, Shape{2}, {32, 2048});
        auto shared_reshaped = std::make_shared<ov::op::v1::Reshape>(shared_gated, reshape_const_output, false);
        auto add_m = std::make_shared<ov::op::v1::Add>(shared_reshaped, moe);

        model = std::make_shared<ov::Model>(add_m, ov::ParameterVector{hidden_states, routing_weights, routing_idx});
        manager.register_pass<FuseMOESharedExpert>();
    }
    {
        // Expected: MOE with 10 inputs (shared expert weights absorbed)
        auto hidden_states = std::make_shared<ov::op::v0::Parameter>(element::f16, Shape{32, 2048});
        auto routing_weights = std::make_shared<ov::op::v0::Parameter>(element::f16, Shape{128, 1, 32, 1});
        auto routing_idx = std::make_shared<ov::op::v0::Parameter>(element::i32, Shape{32, 8});

        auto hidden_states_f32 = std::make_shared<ov::op::v0::Convert>(hidden_states, element::f32);

        // Same decompression chains as input — FuseMOESharedExpert doesn't modify them
        auto gate_decompressed = create_decompression_chain(
            element::u4, {128, 768, 16, 128}, {128, 768, 16, 1}, {128, 768, 16, 1}, {128, 768, 2048});
        auto up_decompressed = create_decompression_chain(
            element::u4, {128, 768, 16, 128}, {128, 768, 16, 1}, {128, 768, 16, 1}, {128, 768, 2048});
        auto down_decompressed = create_decompression_chain(
            element::u4, {128, 2048, 6, 128}, {128, 2048, 6, 1}, {128, 2048, 6, 1}, {128, 2048, 768});

        auto sh_gate_decompressed = create_decompression_chain(
            element::u4, {768, 16, 128}, {768, 16, 1}, {768, 16, 1}, {768, 2048}, 2.0f, 1.0f, 0.02f);
        auto sh_up_decompressed = create_decompression_chain(
            element::u4, {768, 16, 128}, {768, 16, 1}, {768, 16, 1}, {768, 2048}, 2.0f, 1.0f, 0.02f);
        auto sh_down_decompressed = create_decompression_chain(
            element::u4, {2048, 6, 128}, {2048, 6, 1}, {2048, 6, 1}, {2048, 768}, 2.0f, 1.0f, 0.02f);

        auto gate_gate_wei = std::make_shared<ov::op::v0::Constant>(element::f32, Shape{2048, 1}, std::vector<float>(2048, 1.0f));

        ov::op::internal::MOE::Config config;
        config.expert_type = ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU;
        auto moe_expected = std::make_shared<ov::op::internal::MOE>(
            ov::OutputVector{hidden_states_f32, routing_weights, routing_idx,
                             gate_decompressed, up_decompressed, down_decompressed,
                             sh_gate_decompressed, sh_up_decompressed, sh_down_decompressed,
                             gate_gate_wei}, config);

        model_ref = std::make_shared<ov::Model>(moe_expected, ov::ParameterVector{hidden_states, routing_weights, routing_idx});
    }
}

TEST_F(TransformationTestsF, FuseMOESharedExpertWithoutGating) {
    disable_rt_info_check();
    {
        // Same as above but without sigmoid gating (shared_down output goes through Add directly)
        auto hidden_states = std::make_shared<ov::op::v0::Parameter>(element::f16, Shape{32, 2048});
        auto routing_weights = std::make_shared<ov::op::v0::Parameter>(element::f16, Shape{128, 1, 32, 1});
        auto routing_idx = std::make_shared<ov::op::v0::Parameter>(element::i32, Shape{32, 8});

        auto gate_decompressed = create_decompression_chain(
            element::u4, {128, 768, 16, 128}, {128, 768, 16, 1}, {128, 768, 16, 1}, {128, 768, 2048});
        auto up_decompressed = create_decompression_chain(
            element::u4, {128, 768, 16, 128}, {128, 768, 16, 1}, {128, 768, 16, 1}, {128, 768, 2048});
        auto down_decompressed = create_decompression_chain(
            element::u4, {128, 2048, 6, 128}, {128, 2048, 6, 1}, {128, 2048, 6, 1}, {128, 2048, 768});

        ov::op::internal::MOE::Config config;
        config.expert_type = ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU;
        auto hidden_states_f32 = std::make_shared<ov::op::v0::Convert>(hidden_states, element::f32);
        auto moe = std::make_shared<ov::op::internal::MOE>(
            ov::OutputVector{hidden_states_f32, routing_weights, routing_idx,
                             gate_decompressed, up_decompressed, down_decompressed}, config);

        auto reshape_const_hs = op::v0::Constant::create(element::i64, Shape{2}, {32, 2048});
        auto hidden_states_reshaped = std::make_shared<ov::op::v1::Reshape>(hidden_states_f32, reshape_const_hs, false);

        auto sh_gate_decompressed = create_decompression_chain(
            element::u4, {768, 16, 128}, {768, 16, 1}, {768, 16, 1}, {768, 2048}, 2.0f, 1.0f, 0.02f);
        auto sh_up_decompressed = create_decompression_chain(
            element::u4, {768, 16, 128}, {768, 16, 1}, {768, 16, 1}, {768, 2048}, 2.0f, 1.0f, 0.02f);
        auto sh_down_decompressed = create_decompression_chain(
            element::u4, {2048, 6, 128}, {2048, 6, 1}, {2048, 6, 1}, {2048, 768}, 2.0f, 1.0f, 0.02f);

        auto shared_gate_m = std::make_shared<ov::op::v0::MatMul>(hidden_states_reshaped, sh_gate_decompressed, false, true);
        auto shared_swish_m = std::make_shared<ov::op::v4::Swish>(shared_gate_m);
        auto shared_up_m = std::make_shared<ov::op::v0::MatMul>(hidden_states_reshaped, sh_up_decompressed, false, true);
        auto shared_mul_m = std::make_shared<ov::op::v1::Multiply>(shared_swish_m, shared_up_m);
        auto shared_down_m = std::make_shared<ov::op::v0::MatMul>(shared_mul_m, sh_down_decompressed, false, true);

        // No sigmoid gating — shared_down goes directly to Add
        auto add_m = std::make_shared<ov::op::v1::Add>(moe, shared_down_m);

        model = std::make_shared<ov::Model>(add_m, ov::ParameterVector{hidden_states, routing_weights, routing_idx});
        manager.register_pass<FuseMOESharedExpert>();
    }
    {
        auto hidden_states = std::make_shared<ov::op::v0::Parameter>(element::f16, Shape{32, 2048});
        auto routing_weights = std::make_shared<ov::op::v0::Parameter>(element::f16, Shape{128, 1, 32, 1});
        auto routing_idx = std::make_shared<ov::op::v0::Parameter>(element::i32, Shape{32, 8});

        auto hidden_states_f32 = std::make_shared<ov::op::v0::Convert>(hidden_states, element::f32);

        auto gate_decompressed = create_decompression_chain(
            element::u4, {128, 768, 16, 128}, {128, 768, 16, 1}, {128, 768, 16, 1}, {128, 768, 2048});
        auto up_decompressed = create_decompression_chain(
            element::u4, {128, 768, 16, 128}, {128, 768, 16, 1}, {128, 768, 16, 1}, {128, 768, 2048});
        auto down_decompressed = create_decompression_chain(
            element::u4, {128, 2048, 6, 128}, {128, 2048, 6, 1}, {128, 2048, 6, 1}, {128, 2048, 768});

        auto sh_gate_decompressed = create_decompression_chain(
            element::u4, {768, 16, 128}, {768, 16, 1}, {768, 16, 1}, {768, 2048}, 2.0f, 1.0f, 0.02f);
        auto sh_up_decompressed = create_decompression_chain(
            element::u4, {768, 16, 128}, {768, 16, 1}, {768, 16, 1}, {768, 2048}, 2.0f, 1.0f, 0.02f);
        auto sh_down_decompressed = create_decompression_chain(
            element::u4, {2048, 6, 128}, {2048, 6, 1}, {2048, 6, 1}, {2048, 768}, 2.0f, 1.0f, 0.02f);

        // Without gating, FuseMOESharedExpert inserts a dummy gate_gate constant (f16 to match ConvertMOEToMOECompressed)
        auto dummy_gate_gate = ov::op::v0::Constant::create(element::f16, Shape{2048, 1}, std::vector<float>(2048, 0.0f));

        ov::op::internal::MOE::Config config;
        config.expert_type = ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU;
        auto moe_expected = std::make_shared<ov::op::internal::MOE>(
            ov::OutputVector{hidden_states_f32, routing_weights, routing_idx,
                             gate_decompressed, up_decompressed, down_decompressed,
                             sh_gate_decompressed, sh_up_decompressed, sh_down_decompressed,
                             dummy_gate_gate}, config);

        model_ref = std::make_shared<ov::Model>(moe_expected, ov::ParameterVector{hidden_states, routing_weights, routing_idx});
    }
}

// Test FuseMOESharedExpert with symmetric quantization (Convert -> Multiply, no Subtract)
TEST_F(TransformationTestsF, FuseMOESharedExpertSymmetricWithGating) {
    disable_rt_info_check();
    {
        auto hidden_states = std::make_shared<ov::op::v0::Parameter>(element::f16, Shape{32, 2048});
        auto routing_weights = std::make_shared<ov::op::v0::Parameter>(element::f16, Shape{128, 1, 32, 1});
        auto routing_idx = std::make_shared<ov::op::v0::Parameter>(element::i32, Shape{32, 8});

        // MOE expert sym-quant decompression chains (no Subtract)
        auto gate_decompressed = create_sym_decompression_chain(
            element::i4, {128, 768, 16, 128}, {128, 768, 16, 1}, {128, 768, 2048});
        auto up_decompressed = create_sym_decompression_chain(
            element::i4, {128, 768, 16, 128}, {128, 768, 16, 1}, {128, 768, 2048});
        auto down_decompressed = create_sym_decompression_chain(
            element::i4, {128, 2048, 6, 128}, {128, 2048, 6, 1}, {128, 2048, 768});

        ov::op::internal::MOE::Config config;
        config.expert_type = ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU;
        auto hidden_states_f32 = std::make_shared<ov::op::v0::Convert>(hidden_states, element::f32);
        auto moe = std::make_shared<ov::op::internal::MOE>(
            ov::OutputVector{hidden_states_f32, routing_weights, routing_idx,
                             gate_decompressed, up_decompressed, down_decompressed}, config);

        // Shared expert with sym-quant decompression chains
        auto reshape_const_hs = op::v0::Constant::create(element::i64, Shape{2}, {32, 2048});
        auto hidden_states_reshaped = std::make_shared<ov::op::v1::Reshape>(hidden_states_f32, reshape_const_hs, false);

        auto sh_gate_decompressed = create_sym_decompression_chain(
            element::i4, {768, 16, 128}, {768, 16, 1}, {768, 2048}, 2.0f, 0.02f);
        auto sh_up_decompressed = create_sym_decompression_chain(
            element::i4, {768, 16, 128}, {768, 16, 1}, {768, 2048}, 2.0f, 0.02f);
        auto sh_down_decompressed = create_sym_decompression_chain(
            element::i4, {2048, 6, 128}, {2048, 6, 1}, {2048, 768}, 2.0f, 0.02f);

        auto shared_gate_m = std::make_shared<ov::op::v0::MatMul>(hidden_states_reshaped, sh_gate_decompressed, false, true);
        auto shared_swish_m = std::make_shared<ov::op::v4::Swish>(shared_gate_m);
        auto shared_up_m = std::make_shared<ov::op::v0::MatMul>(hidden_states_reshaped, sh_up_decompressed, false, true);
        auto shared_mul_m = std::make_shared<ov::op::v1::Multiply>(shared_swish_m, shared_up_m);
        auto shared_down_m = std::make_shared<ov::op::v0::MatMul>(shared_mul_m, sh_down_decompressed, false, true);

        // Sigmoid gating
        auto gate_gate_wei = std::make_shared<ov::op::v0::Constant>(element::f32, Shape{2048, 1}, std::vector<float>(2048, 1.0f));
        auto gate_gate_mm = std::make_shared<ov::op::v0::MatMul>(hidden_states_reshaped, gate_gate_wei);
        auto gate_sigmoid = std::make_shared<ov::op::v0::Sigmoid>(gate_gate_mm);
        auto shared_gated = std::make_shared<ov::op::v1::Multiply>(gate_sigmoid, shared_down_m);

        auto reshape_const_output = op::v0::Constant::create(element::i64, Shape{2}, {32, 2048});
        auto shared_reshaped = std::make_shared<ov::op::v1::Reshape>(shared_gated, reshape_const_output, false);
        auto add_m = std::make_shared<ov::op::v1::Add>(moe, shared_reshaped);

        model = std::make_shared<ov::Model>(add_m, ov::ParameterVector{hidden_states, routing_weights, routing_idx});
        manager.register_pass<FuseMOESharedExpert>();
    }
    {
        auto hidden_states = std::make_shared<ov::op::v0::Parameter>(element::f16, Shape{32, 2048});
        auto routing_weights = std::make_shared<ov::op::v0::Parameter>(element::f16, Shape{128, 1, 32, 1});
        auto routing_idx = std::make_shared<ov::op::v0::Parameter>(element::i32, Shape{32, 8});

        auto hidden_states_f32 = std::make_shared<ov::op::v0::Convert>(hidden_states, element::f32);

        auto gate_decompressed = create_sym_decompression_chain(
            element::i4, {128, 768, 16, 128}, {128, 768, 16, 1}, {128, 768, 2048});
        auto up_decompressed = create_sym_decompression_chain(
            element::i4, {128, 768, 16, 128}, {128, 768, 16, 1}, {128, 768, 2048});
        auto down_decompressed = create_sym_decompression_chain(
            element::i4, {128, 2048, 6, 128}, {128, 2048, 6, 1}, {128, 2048, 768});

        auto sh_gate_decompressed = create_sym_decompression_chain(
            element::i4, {768, 16, 128}, {768, 16, 1}, {768, 2048}, 2.0f, 0.02f);
        auto sh_up_decompressed = create_sym_decompression_chain(
            element::i4, {768, 16, 128}, {768, 16, 1}, {768, 2048}, 2.0f, 0.02f);
        auto sh_down_decompressed = create_sym_decompression_chain(
            element::i4, {2048, 6, 128}, {2048, 6, 1}, {2048, 768}, 2.0f, 0.02f);

        auto gate_gate_wei = std::make_shared<ov::op::v0::Constant>(element::f32, Shape{2048, 1}, std::vector<float>(2048, 1.0f));

        ov::op::internal::MOE::Config config;
        config.expert_type = ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU;
        auto moe_expected = std::make_shared<ov::op::internal::MOE>(
            ov::OutputVector{hidden_states_f32, routing_weights, routing_idx,
                             gate_decompressed, up_decompressed, down_decompressed,
                             sh_gate_decompressed, sh_up_decompressed, sh_down_decompressed,
                             gate_gate_wei}, config);

        model_ref = std::make_shared<ov::Model>(moe_expected, ov::ParameterVector{hidden_states, routing_weights, routing_idx});
    }
}

// Test FuseMOESharedExpert with MOECompressed input (sigmoid gating)
TEST_F(TransformationTestsF, FuseMOECompressedSharedExpertWithSigmoidGating) {
    disable_rt_info_check();
    {
        // tokens:32, hidden_size:2048, inter_size:768, experts:128, topk:8
        auto hidden_states = std::make_shared<ov::op::v0::Parameter>(element::f16, Shape{32, 2048});
        auto routing_weights = std::make_shared<ov::op::v0::Parameter>(element::f16, Shape{128, 1, 32, 1});
        auto routing_idx = std::make_shared<ov::op::v0::Parameter>(element::i32, Shape{32, 8});

        // MOE compressed weights (no decompression chains)
        auto wei_gate = op::v0::Constant::create(element::u4, Shape{128, 768, 16, 128}, {1});
        auto scale_gate = op::v0::Constant::create(element::f16, Shape{128, 16, 768}, {0.01f});
        auto zp_gate = op::v0::Constant::create(element::u4, Shape{128, 16, 768}, {0});
        auto wei_up = op::v0::Constant::create(element::u4, Shape{128, 768, 16, 128}, {1});
        auto scale_up = op::v0::Constant::create(element::f16, Shape{128, 16, 768}, {0.01f});
        auto zp_up = op::v0::Constant::create(element::u4, Shape{128, 16, 768}, {0});
        auto wei_down = op::v0::Constant::create(element::u4, Shape{128, 2048, 6, 128}, {1});
        auto scale_down = op::v0::Constant::create(element::f16, Shape{128, 6, 2048}, {0.01f});
        auto zp_down = op::v0::Constant::create(element::u4, Shape{128, 6, 2048}, {0});

        ov::op::internal::MOECompressed::Config config;
        config.expert_type = ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU;
        config.hidden_size = 2048;
        config.inter_size = 768;
        config.num_expert = 128;
        config.group_size = 128;
        config.top_k = 8;
        config.out_type = ov::element::f32;
        auto moe = std::make_shared<ov::op::internal::MOECompressed>(
            ov::OutputVector{hidden_states, routing_weights, routing_idx,
                             wei_gate, scale_gate, zp_gate, wei_up, scale_up, zp_up,
                             wei_down, scale_down, zp_down}, config);

        // Shared expert subgraph with decompression chains (outputs f32)
        auto hidden_states_f32 = std::make_shared<ov::op::v0::Convert>(hidden_states, element::f32);
        auto reshape_const_hs = op::v0::Constant::create(element::i64, Shape{2}, {32, 2048});
        auto hidden_states_reshaped = std::make_shared<ov::op::v1::Reshape>(hidden_states_f32, reshape_const_hs, false);

        auto sh_gate_decompressed = create_decompression_chain(
            element::u4, {768, 16, 128}, {768, 16, 1}, {768, 16, 1}, {768, 2048}, 2.0f, 1.0f, 0.02f);
        auto sh_up_decompressed = create_decompression_chain(
            element::u4, {768, 16, 128}, {768, 16, 1}, {768, 16, 1}, {768, 2048}, 2.0f, 1.0f, 0.02f);
        auto sh_down_decompressed = create_decompression_chain(
            element::u4, {2048, 6, 128}, {2048, 6, 1}, {2048, 6, 1}, {2048, 768}, 2.0f, 1.0f, 0.02f);

        auto shared_gate_m = std::make_shared<ov::op::v0::MatMul>(hidden_states_reshaped, sh_gate_decompressed, false, true);
        auto shared_swish_m = std::make_shared<ov::op::v4::Swish>(shared_gate_m);
        auto shared_up_m = std::make_shared<ov::op::v0::MatMul>(hidden_states_reshaped, sh_up_decompressed, false, true);
        auto shared_mul_m = std::make_shared<ov::op::v1::Multiply>(shared_swish_m, shared_up_m);
        auto shared_down_m = std::make_shared<ov::op::v0::MatMul>(shared_mul_m, sh_down_decompressed, false, true);

        // Sigmoid gating
        auto gate_gate_wei = std::make_shared<ov::op::v0::Constant>(element::f32, Shape{2048, 1}, std::vector<float>(2048, 1.0f));
        auto gate_gate_mm = std::make_shared<ov::op::v0::MatMul>(hidden_states_reshaped, gate_gate_wei);
        auto gate_sigmoid = std::make_shared<ov::op::v0::Sigmoid>(gate_gate_mm);
        auto shared_gated = std::make_shared<ov::op::v1::Multiply>(gate_sigmoid, shared_down_m);

        auto reshape_const_output = op::v0::Constant::create(element::i64, Shape{2}, {32, 2048});
        auto shared_reshaped = std::make_shared<ov::op::v1::Reshape>(shared_gated, reshape_const_output, false);
        auto add_m = std::make_shared<ov::op::v1::Add>(shared_reshaped, moe);

        model = std::make_shared<ov::Model>(add_m, ov::ParameterVector{hidden_states, routing_weights, routing_idx});
        manager.register_pass<FuseMOESharedExpert>();
    }
    {
        // Expected: MOECompressed with shared expert weights absorbed
        auto hidden_states = std::make_shared<ov::op::v0::Parameter>(element::f16, Shape{32, 2048});
        auto routing_weights = std::make_shared<ov::op::v0::Parameter>(element::f16, Shape{128, 1, 32, 1});
        auto routing_idx = std::make_shared<ov::op::v0::Parameter>(element::i32, Shape{32, 8});

        auto wei_gate = op::v0::Constant::create(element::u4, Shape{128, 768, 16, 128}, {1});
        auto scale_gate = op::v0::Constant::create(element::f16, Shape{128, 16, 768}, {0.01f});
        auto zp_gate = op::v0::Constant::create(element::u4, Shape{128, 16, 768}, {0});
        auto wei_up = op::v0::Constant::create(element::u4, Shape{128, 768, 16, 128}, {1});
        auto scale_up = op::v0::Constant::create(element::f16, Shape{128, 16, 768}, {0.01f});
        auto zp_up = op::v0::Constant::create(element::u4, Shape{128, 16, 768}, {0});
        auto wei_down = op::v0::Constant::create(element::u4, Shape{128, 2048, 6, 128}, {1});
        auto scale_down = op::v0::Constant::create(element::f16, Shape{128, 6, 2048}, {0.01f});
        auto zp_down = op::v0::Constant::create(element::u4, Shape{128, 6, 2048}, {0});

        auto sh_gate_decompressed = create_decompression_chain(
            element::u4, {768, 16, 128}, {768, 16, 1}, {768, 16, 1}, {768, 2048}, 2.0f, 1.0f, 0.02f);
        auto sh_up_decompressed = create_decompression_chain(
            element::u4, {768, 16, 128}, {768, 16, 1}, {768, 16, 1}, {768, 2048}, 2.0f, 1.0f, 0.02f);
        auto sh_down_decompressed = create_decompression_chain(
            element::u4, {2048, 6, 128}, {2048, 6, 1}, {2048, 6, 1}, {2048, 768}, 2.0f, 1.0f, 0.02f);

        auto gate_gate_wei = std::make_shared<ov::op::v0::Constant>(element::f32, Shape{2048, 1}, std::vector<float>(2048, 1.0f));

        ov::op::internal::MOECompressed::Config config;
        config.expert_type = ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU;
        config.hidden_size = 2048;
        config.inter_size = 768;
        config.num_expert = 128;
        config.group_size = 128;
        config.top_k = 8;
        config.out_type = ov::element::f32;
        auto moe_expected = std::make_shared<ov::op::internal::MOECompressed>(
            ov::OutputVector{hidden_states, routing_weights, routing_idx,
                             wei_gate, scale_gate, zp_gate, wei_up, scale_up, zp_up,
                             wei_down, scale_down, zp_down,
                             sh_gate_decompressed, sh_up_decompressed, sh_down_decompressed,
                             gate_gate_wei}, config);

        model_ref = std::make_shared<ov::Model>(moe_expected, ov::ParameterVector{hidden_states, routing_weights, routing_idx});
    }
}

}  // namespace intel_gpu
}  // namespace test
}  // namespace ov
