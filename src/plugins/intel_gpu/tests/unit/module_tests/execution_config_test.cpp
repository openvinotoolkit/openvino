// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include "intel_gpu/plugin/remote_context.hpp"
#include "intel_gpu/runtime/execution_config.hpp"

#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/op/subtract.hpp"

using namespace cldnn;
using namespace ov::intel_gpu;
using namespace ::tests;

// Build a model with MatMul (compressed weights) + PagedAttention stub.
// PA node triggers is_paged_attention_model detection in apply_model_specific_options.
static std::shared_ptr<ov::Model> make_pa_matmul_model(ov::element::Type weight_type) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 16});

    auto weight_const = ov::op::v0::Constant::create(weight_type, ov::Shape{32, 16}, {1});

    std::shared_ptr<ov::Node> weight_node;
    if (weight_type == ov::element::u4 || weight_type == ov::element::i4 || weight_type == ov::element::u8) {
        auto convert  = std::make_shared<ov::op::v0::Convert>(weight_const, ov::element::f32);
        auto zp_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 1}, {0});
        auto subtract = std::make_shared<ov::op::v1::Subtract>(convert, zp_const);
        auto sc_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 1}, {1});
        auto multiply = std::make_shared<ov::op::v1::Multiply>(subtract, sc_const);
        weight_node = multiply;
    } else {
        weight_node = weight_const;
    }

    auto matmul = std::make_shared<ov::op::v0::MatMul>(input, weight_node, false, true);

    // Minimal PagedAttention node (26 inputs required)
    const size_t hs = 64;
    auto pa_q  = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, hs});
    auto pa_k  = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, hs});
    auto pa_v  = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, hs});
    auto pa_kc = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 1, 16, hs});
    auto pa_vc = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 1, 16, hs});
    auto pa_pl = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1});
    auto pa_sb = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1});
    auto pa_bi = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1});
    auto pa_bb = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1});
    auto pa_sc = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {1.0f});
    auto pa_sw = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    auto pa_al = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{0}, {});
    auto pa_mc = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{});
    auto pa_sa = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1});
    auto pa_rb = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1});
    auto pa_rd = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1, 1});
    auto pa_rt = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{256, hs});
    auto pa_xt = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1});
    auto pa_xb = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{});
    auto pa_xs = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{});
    auto pa_sk = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 1, 1, 1}, {0});
    auto pa_as = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    auto pa_ae = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1});
    auto pa_ai = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1});
    auto pa_ab = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1});
    auto pa_tt = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{0});

    ov::OutputVector pa_args = {pa_q, pa_k, pa_v, pa_kc, pa_vc,
                                pa_pl, pa_sb, pa_bi, pa_bb, pa_sc, pa_sw,
                                pa_al, pa_mc, pa_sa, pa_rb, pa_rd, pa_rt,
                                pa_xt, pa_xb, pa_xs, pa_sk, pa_as, pa_ae,
                                pa_ai, pa_ab, pa_tt};
    auto pa_node = std::make_shared<ov::op::PagedAttentionExtension>(pa_args);

    return std::make_shared<ov::Model>(
        ov::OutputVector{matmul, pa_node->output(0)},
        ov::ParameterVector{input, pa_q, pa_k, pa_v, pa_kc, pa_vc,
                            pa_pl, pa_sb, pa_bi, pa_bb, pa_mc, pa_sa,
                            pa_rb, pa_rd, pa_rt, pa_xt, pa_xb, pa_xs,
                            pa_ae, pa_ai, pa_ab, pa_tt});
}

TEST(execution_config, kv_cache_u4_weights_auto_detect_u4) {
    auto& engine = get_test_engine();
    auto ctx = std::make_shared<RemoteContextImpl>("GPU", std::vector<cldnn::device::ptr>{engine.get_device()});
    auto model = make_pa_matmul_model(ov::element::u4);

    ExecutionConfig config;
    config.finalize(ctx.get(), model.get());

    ASSERT_EQ(config.get_kv_cache_precision(), ov::element::u4);
}

TEST(execution_config, kv_cache_i4_weights_auto_detect_u4) {
    auto& engine = get_test_engine();
    auto ctx = std::make_shared<RemoteContextImpl>("GPU", std::vector<cldnn::device::ptr>{engine.get_device()});
    auto model = make_pa_matmul_model(ov::element::i4);

    ExecutionConfig config;
    config.finalize(ctx.get(), model.get());

    // i4 weights → auto-detect u4, and finalize normalizes i4→u4
    ASSERT_EQ(config.get_kv_cache_precision(), ov::element::u4);
}

TEST(execution_config, kv_cache_u8_weights_no_u4) {
    auto& engine = get_test_engine();
    auto ctx = std::make_shared<RemoteContextImpl>("GPU", std::vector<cldnn::device::ptr>{engine.get_device()});
    auto model = make_pa_matmul_model(ov::element::u8);

    ExecutionConfig config;
    config.finalize(ctx.get(), model.get());

    ASSERT_NE(config.get_kv_cache_precision(), ov::element::u4);
}

TEST(execution_config, kv_cache_f32_weights_no_u4) {
    auto& engine = get_test_engine();
    auto ctx = std::make_shared<RemoteContextImpl>("GPU", std::vector<cldnn::device::ptr>{engine.get_device()});
    auto model = make_pa_matmul_model(ov::element::f32);

    ExecutionConfig config;
    config.finalize(ctx.get(), model.get());

    ASSERT_NE(config.get_kv_cache_precision(), ov::element::u4);
}

TEST(execution_config, kv_cache_user_override_wins) {
    auto& engine = get_test_engine();
    auto ctx = std::make_shared<RemoteContextImpl>("GPU", std::vector<cldnn::device::ptr>{engine.get_device()});
    auto model = make_pa_matmul_model(ov::element::u4);

    ExecutionConfig config;
    config.set_user_property(ov::hint::kv_cache_precision(ov::element::i8));
    config.finalize(ctx.get(), model.get());

    ASSERT_EQ(config.get_kv_cache_precision(), ov::element::i8);
}

TEST(execution_config, kv_cache_i4_normalized_to_u4) {
    auto& engine = get_test_engine();
    auto ctx = std::make_shared<RemoteContextImpl>("GPU", std::vector<cldnn::device::ptr>{engine.get_device()});
    auto model = make_pa_matmul_model(ov::element::f32);

    ExecutionConfig config;
    config.set_user_property(ov::hint::kv_cache_precision(ov::element::i4));
    config.finalize(ctx.get(), model.get());

    ASSERT_EQ(config.get_kv_cache_precision(), ov::element::u4);
}

TEST(execution_config, kv_cache_u8_normalized_to_i8) {
    auto& engine = get_test_engine();
    auto ctx = std::make_shared<RemoteContextImpl>("GPU", std::vector<cldnn::device::ptr>{engine.get_device()});
    auto model = make_pa_matmul_model(ov::element::f32);

    ExecutionConfig config;
    config.set_user_property(ov::hint::kv_cache_precision(ov::element::u8));
    config.finalize(ctx.get(), model.get());

    ASSERT_EQ(config.get_kv_cache_precision(), ov::element::i8);
}

TEST(execution_config, kv_cache_4bit_by_token_throws) {
    auto& engine = get_test_engine();
    auto ctx = std::make_shared<RemoteContextImpl>("GPU", std::vector<cldnn::device::ptr>{engine.get_device()});
    auto model = make_pa_matmul_model(ov::element::u4);

    ExecutionConfig config;
    config.set_user_property(ov::hint::kv_cache_precision(ov::element::u4));
    config.set_user_property(ov::internal::key_cache_quant_mode(ov::internal::CacheQuantMode::BY_TOKEN));

    ASSERT_ANY_THROW(config.finalize(ctx.get(), model.get()));
}
