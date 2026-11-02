// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <limits>
#include <memory>
#include <transformations/cpu_opset/common/op/sdpa.hpp>
#include <transformations/cpu_opset/common/pass/stateful_sdpa_fusion.hpp>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/unsqueeze.hpp"

using namespace testing;
using namespace ov;
using namespace ov::intel_cpu;

namespace {
// Grouped-query attention over a KV cache, with the key/value heads repeated the way the PyTorch frontend converts
// scaled_dot_product_attention(enable_gqa=True) with static head counts: [B, Hk, S, D] -> Unsqueeze(-3)
// -> Broadcast(BIDIRECTIONAL, Concat(ShapeOf[:-2], G, ShapeOf[-2], ShapeOf[-1]))
// -> Reshape(Concat(ShapeOf[:-3], Hq, ShapeOf[-2], ShapeOf[-1])). The ShapeOf adds a consumer to the KV-cache Concat,
// which the fusion must accept.
std::shared_ptr<ov::Model> make_gqa_kv_cache_model(int64_t num_q_heads, int64_t num_kv_heads, bool is_ref) {
    constexpr int64_t head_size = 64;
    auto q = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, num_q_heads, -1, head_size});
    auto k = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, num_kv_heads, -1, head_size});
    auto v = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, num_kv_heads, -1, head_size});
    auto beam_idx = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{-1});
    const PartialShape kv_shape{-1, num_kv_heads, -1, head_size};
    auto var_k = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{kv_shape, element::f32, "pastk"});
    auto var_v = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{kv_shape, element::f32, "pastv"});
    auto past_k = std::make_shared<ov::op::v6::ReadValue>(k, var_k);
    auto past_v = std::make_shared<ov::op::v6::ReadValue>(v, var_v);

    Output<Node> sdpa, present_k, present_v;
    if (is_ref) {
        ScaledDotProductAttentionWithKVCache::Config config;
        config.fuse_concat = true;
        auto fused =
            std::make_shared<ScaledDotProductAttentionWithKVCache>(OutputVector{q, k, v, beam_idx, past_k, past_v},
                                                                   config);
        sdpa = fused->output(0);
        present_k = fused->output(1);
        present_v = fused->output(2);
    } else {
        auto axis = ov::op::v0::Constant::create(element::i32, {1}, {0});
        present_k = std::make_shared<ov::op::v0::Concat>(
            OutputVector{std::make_shared<ov::op::v8::Gather>(past_k, beam_idx, axis), k},
            2);
        present_v = std::make_shared<ov::op::v0::Concat>(
            OutputVector{std::make_shared<ov::op::v8::Gather>(past_v, beam_idx, axis), v},
            2);
        auto repeat_heads = [&](const Output<Node>& kv) -> Output<Node> {
            auto shape = std::make_shared<ov::op::v3::ShapeOf>(kv, element::i64);
            auto slice_dims = [&](int64_t begin, int64_t end) {
                return std::make_shared<ov::op::v8::Slice>(shape,
                                                           ov::op::v0::Constant::create(element::i64, {1}, {begin}),
                                                           ov::op::v0::Constant::create(element::i64, {1}, {end}),
                                                           ov::op::v0::Constant::create(element::i64, {1}, {1}));
            };
            auto seq_len = slice_dims(-2, -1);
            auto head_dim = slice_dims(-1, std::numeric_limits<int64_t>::max());
            auto target = std::make_shared<ov::op::v0::Concat>(
                OutputVector{slice_dims(0, -2),
                             ov::op::v0::Constant::create(element::i64, {1}, {num_q_heads / num_kv_heads}),
                             seq_len,
                             head_dim},
                0);
            auto output_shape = std::make_shared<ov::op::v0::Concat>(
                OutputVector{slice_dims(0, -3),
                             ov::op::v0::Constant::create(element::i64, {1}, {num_q_heads}),
                             seq_len,
                             head_dim},
                0);
            auto unsqueezed =
                std::make_shared<ov::op::v0::Unsqueeze>(kv, ov::op::v0::Constant::create(element::i64, {1}, {-3}));
            auto broadcasted =
                std::make_shared<ov::op::v3::Broadcast>(unsqueezed, target, ov::op::BroadcastType::BIDIRECTIONAL);
            return std::make_shared<ov::op::v1::Reshape>(broadcasted, output_shape, false);
        };
        sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(q,
                                                                        repeat_heads(present_k),
                                                                        repeat_heads(present_v),
                                                                        false);
    }
    auto add = std::make_shared<ov::op::v1::Add>(sdpa, ov::op::v0::Constant::create(element::f32, {1}, {1.0F}));
    SinkVector sinks{std::make_shared<ov::op::v6::Assign>(present_k, var_k),
                     std::make_shared<ov::op::v6::Assign>(present_v, var_v)};
    return std::make_shared<ov::Model>(ResultVector{std::make_shared<ov::op::v0::Result>(add)},
                                       sinks,
                                       ParameterVector{q, k, v, beam_idx},
                                       "GqaKvCache");
}
}  // namespace

TEST_F(TransformationTestsF, StateConcatSDPAGroupedQueryShapeOfBroadcast) {
#if defined(OPENVINO_ARCH_X86_64) && (defined(__ANDROID__) || defined(ANDROID))
    test_skipped = true;
    GTEST_SKIP() << "Skipping StateConcatSDPAGroupedQueryShapeOfBroadcast test on Android X64";
#endif
    model = make_gqa_kv_cache_model(8, 2, false);
    model_ref = make_gqa_kv_cache_model(8, 2, true);
    manager.register_pass<StatefulSDPAFusion>();
}

TEST_F(TransformationTestsF, StateConcatSDPAMultiQueryShapeOfBroadcast) {
#if defined(OPENVINO_ARCH_X86_64) && (defined(__ANDROID__) || defined(ANDROID))
    test_skipped = true;
    GTEST_SKIP() << "Skipping StateConcatSDPAMultiQueryShapeOfBroadcast test on Android X64";
#endif
    model = make_gqa_kv_cache_model(8, 1, false);
    model_ref = make_gqa_kv_cache_model(8, 1, true);
    manager.register_pass<StatefulSDPAFusion>();
}
