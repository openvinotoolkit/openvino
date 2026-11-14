// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <limits>
#include <memory>
#include <transformations/cpu_opset/common/op/sdpa.hpp>
#include <transformations/cpu_opset/common/pass/stateful_sdpa_fusion.hpp>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"

using namespace testing;
using namespace ov;
using namespace ov::intel_cpu;

namespace {
// Grouped-query attention over a KV cache, with the key/value heads repeated the way the PyTorch frontend converts
// scaled_dot_product_attention(enable_gqa=True) with static head counts: [B, Hk, S, D] -> Unsqueeze(-3)
// -> Broadcast(BIDIRECTIONAL, Concat(ShapeOf[:-2], G, ShapeOf[-2], ShapeOf[-1]))
// -> Reshape(Concat(ShapeOf[:-3], Hq, ShapeOf[-2], ShapeOf[-1])). Key/value with as many heads as the query are not
// repeated. The ShapeOf adds a consumer to the KV-cache Concat, which the fusion must accept.
struct KvCacheCase {
    int64_t rank = 4;  // 5 adds a static prefix dimension: [B, X, H, L, D]
    int64_t q_heads = 8;
    int64_t k_heads = 2;
    int64_t v_heads = 2;
    bool causal = false;
    bool transposed = false;  // [B, L, H, D] inputs, transposed to [B, H, L, D] for the SDPA (rank 4 only)
    enum class Mask { None, Input, ScalarZero } mask = Mask::None;
    enum class Scale { None, Constant, F16Constant, ConvertLikeConstant, Input } scale = Scale::None;
    float scale_value = 0.125F;
};

std::shared_ptr<ov::Model> make_gqa_kv_cache_model(const KvCacheCase& c, bool is_ref) {
    constexpr int64_t head_size = 64;
    auto input_shape = [&](int64_t heads) {
        std::vector<Dimension> dims{-1};
        if (c.rank == 5) {
            dims.emplace_back(3);
        }
        if (c.transposed) {
            dims.insert(dims.end(), {-1, heads, head_size});
        } else {
            dims.insert(dims.end(), {heads, -1, head_size});
        }
        return PartialShape(dims);
    };
    ParameterVector params;
    auto make_param = [&](const element::Type& type, const PartialShape& shape) {
        params.push_back(std::make_shared<ov::op::v0::Parameter>(type, shape));
        return params.back();
    };
    auto q = make_param(element::f32, input_shape(c.q_heads));
    auto k = make_param(element::f32, input_shape(c.k_heads));
    auto v = make_param(element::f32, input_shape(c.v_heads));
    auto beam_idx = make_param(element::i32, PartialShape{-1});
    auto var_k = std::make_shared<ov::op::util::Variable>(
        ov::op::util::VariableInfo{input_shape(c.k_heads), element::f32, "pastk"});
    auto var_v = std::make_shared<ov::op::util::Variable>(
        ov::op::util::VariableInfo{input_shape(c.v_heads), element::f32, "pastv"});
    auto past_k = std::make_shared<ov::op::v6::ReadValue>(k, var_k);
    auto past_v = std::make_shared<ov::op::v6::ReadValue>(v, var_v);

    // Mask and scale inputs, laid out as the PyTorch frontend does: a scale without a mask gets a scalar zero mask.
    OutputVector mask_and_scale;
    const auto mask = c.scale != KvCacheCase::Scale::None && c.mask == KvCacheCase::Mask::None
                          ? KvCacheCase::Mask::ScalarZero
                          : c.mask;
    if (mask == KvCacheCase::Mask::Input) {
        mask_and_scale.push_back(make_param(element::f32, PartialShape{-1, 1, -1, -1}));
    } else if (mask == KvCacheCase::Mask::ScalarZero) {
        mask_and_scale.push_back(
            std::make_shared<ov::op::v1::ConvertLike>(ov::op::v0::Constant::create(element::f32, {}, {0.0F}), q));
    }
    if (c.scale == KvCacheCase::Scale::Constant) {
        mask_and_scale.push_back(ov::op::v0::Constant::create(element::f32, {}, {c.scale_value}));
    } else if (c.scale == KvCacheCase::Scale::F16Constant) {
        mask_and_scale.push_back(
            std::make_shared<ov::op::v0::Convert>(ov::op::v0::Constant::create(element::f16, {}, {c.scale_value}),
                                                  element::f32));
    } else if (c.scale == KvCacheCase::Scale::ConvertLikeConstant) {
        mask_and_scale.push_back(
            std::make_shared<ov::op::v1::ConvertLike>(ov::op::v0::Constant::create(element::f32, {}, {c.scale_value}),
                                                      q));
    } else if (c.scale == KvCacheCase::Scale::Input) {
        mask_and_scale.push_back(make_param(element::f32, PartialShape{}));
    }

    Output<Node> sdpa, present_k, present_v;
    if (is_ref) {
        ScaledDotProductAttentionWithKVCache::Config config;
        config.fuse_concat = true;
        config.is_causal = c.causal;
        if (c.transposed) {
            config.permute_axes = {0, 2, 1, 3};
        }
        OutputVector args{q, k, v};
        args.insert(args.end(), mask_and_scale.begin(), mask_and_scale.end());
        args.insert(args.end(), {beam_idx, past_k, past_v});
        auto fused = std::make_shared<ScaledDotProductAttentionWithKVCache>(args, config);
        sdpa = fused->output(0);
        present_k = fused->output(1);
        present_v = fused->output(2);
    } else {
        auto axis = ov::op::v0::Constant::create(element::i32, {1}, {0});
        present_k = std::make_shared<ov::op::v0::Concat>(
            OutputVector{std::make_shared<ov::op::v8::Gather>(past_k, beam_idx, axis), k},
            c.transposed ? 1 : c.rank - 2);
        present_v = std::make_shared<ov::op::v0::Concat>(
            OutputVector{std::make_shared<ov::op::v8::Gather>(past_v, beam_idx, axis), v},
            c.transposed ? 1 : c.rank - 2);
        auto repeat_heads = [&](const Output<Node>& kv, int64_t kv_heads) -> Output<Node> {
            if (kv_heads == c.q_heads) {
                return kv;
            }
            auto shape = std::make_shared<ov::op::v3::ShapeOf>(kv, element::i64);
            auto slice_dims = [&](int64_t begin, int64_t end) {
                return std::make_shared<ov::op::v8::Slice>(shape,
                                                           ov::op::v0::Constant::create(element::i64, {1}, {begin}),
                                                           ov::op::v0::Constant::create(element::i64, {1}, {end}),
                                                           ov::op::v0::Constant::create(element::i64, {1}, {1}));
            };
            const auto group = ov::op::v0::Constant::create(element::i64, {1}, {c.q_heads / kv_heads});
            const auto q_heads = ov::op::v0::Constant::create(element::i64, {1}, {c.q_heads});
            auto head_dim = slice_dims(-1, std::numeric_limits<int64_t>::max());
            if (c.transposed) {
                // [B, L, Hk, D] -> [B, L, Hk, 1, D] -> [B, L, Hk, G, D] -> [B, L, Hq, D]
                auto target = std::make_shared<ov::op::v0::Concat>(OutputVector{slice_dims(0, -1), group, head_dim}, 0);
                auto output_shape =
                    std::make_shared<ov::op::v0::Concat>(OutputVector{slice_dims(0, -2), q_heads, head_dim}, 0);
                auto unsqueezed =
                    std::make_shared<ov::op::v0::Unsqueeze>(kv, ov::op::v0::Constant::create(element::i64, {1}, {-2}));
                auto broadcasted =
                    std::make_shared<ov::op::v3::Broadcast>(unsqueezed, target, ov::op::BroadcastType::BIDIRECTIONAL);
                return std::make_shared<ov::op::v1::Reshape>(broadcasted, output_shape, false);
            }
            auto seq_len = slice_dims(-2, -1);
            auto target =
                std::make_shared<ov::op::v0::Concat>(OutputVector{slice_dims(0, -2), group, seq_len, head_dim}, 0);
            auto output_shape =
                std::make_shared<ov::op::v0::Concat>(OutputVector{slice_dims(0, -3), q_heads, seq_len, head_dim}, 0);
            auto unsqueezed =
                std::make_shared<ov::op::v0::Unsqueeze>(kv, ov::op::v0::Constant::create(element::i64, {1}, {-3}));
            auto broadcasted =
                std::make_shared<ov::op::v3::Broadcast>(unsqueezed, target, ov::op::BroadcastType::BIDIRECTIONAL);
            return std::make_shared<ov::op::v1::Reshape>(broadcasted, output_shape, false);
        };
        OutputVector inputs{q, repeat_heads(present_k, c.k_heads), repeat_heads(present_v, c.v_heads)};
        if (c.transposed) {
            const auto order = ov::op::v0::Constant::create(element::i64, {4}, {0, 2, 1, 3});
            for (size_t i = 0; i < 3; ++i) {
                inputs[i] = std::make_shared<ov::op::v1::Transpose>(inputs[i], order);
            }
        }
        inputs.insert(inputs.end(), mask_and_scale.begin(), mask_and_scale.end());
        sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(inputs, c.causal);
    }
    auto add = std::make_shared<ov::op::v1::Add>(sdpa, ov::op::v0::Constant::create(element::f32, {1}, {1.0F}));
    SinkVector sinks{std::make_shared<ov::op::v6::Assign>(present_k, var_k),
                     std::make_shared<ov::op::v6::Assign>(present_v, var_v)};
    return std::make_shared<ov::Model>(ResultVector{std::make_shared<ov::op::v0::Result>(add)},
                                       sinks,
                                       params,
                                       "GqaKvCache");
}
}  // namespace

#if defined(OPENVINO_ARCH_X86_64) && (defined(__ANDROID__) || defined(ANDROID))
#    define SKIP_ON_ANDROID_X64() \
        test_skipped = true;      \
        GTEST_SKIP() << "Skipping StatefulSDPAFusion tests on Android X64"
#else
#    define SKIP_ON_ANDROID_X64()
#endif

TEST_F(TransformationTestsF, StateConcatSDPAGroupedQueryShapeOfBroadcast) {
    SKIP_ON_ANDROID_X64();
    model = make_gqa_kv_cache_model({}, false);
    model_ref = make_gqa_kv_cache_model({}, true);
    manager.register_pass<StatefulSDPAFusion>();
}

TEST_F(TransformationTestsF, StateConcatSDPAMultiQueryShapeOfBroadcast) {
    SKIP_ON_ANDROID_X64();
    KvCacheCase c;
    c.k_heads = c.v_heads = 1;
    model = make_gqa_kv_cache_model(c, false);
    model_ref = make_gqa_kv_cache_model(c, true);
    manager.register_pass<StatefulSDPAFusion>();
}

// Cases the fused node supports: the fusion still applies.
class StateConcatSDPAFusedTest : public TransformationTestsF, public WithParamInterface<KvCacheCase> {};

TEST_P(StateConcatSDPAFusedTest, Fused) {
    SKIP_ON_ANDROID_X64();
    // is_causal and permute_axes are attributes of the fused node
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    model = make_gqa_kv_cache_model(GetParam(), false);
    model_ref = make_gqa_kv_cache_model(GetParam(), true);
    manager.register_pass<StatefulSDPAFusion>();
}

namespace {
KvCacheCase transposed_case(int64_t k_heads, int64_t v_heads) {
    KvCacheCase c;
    c.transposed = true;
    c.k_heads = k_heads;
    c.v_heads = v_heads;
    return c;
}

KvCacheCase fused_case(bool causal, KvCacheCase::Mask mask, KvCacheCase::Scale scale, int64_t heads = 2) {
    KvCacheCase c;
    c.causal = causal;
    c.mask = mask;
    c.scale = scale;
    c.k_heads = c.v_heads = heads;
    if (heads == -1) {
        c.q_heads = -1;
    }
    return c;
}
}  // namespace

INSTANTIATE_TEST_SUITE_P(
    TransformationTests,
    StateConcatSDPAFusedTest,
    ::testing::Values(fused_case(true, KvCacheCase::Mask::None, KvCacheCase::Scale::None),
                      fused_case(false, KvCacheCase::Mask::Input, KvCacheCase::Scale::Constant),
                      fused_case(false, KvCacheCase::Mask::Input, KvCacheCase::Scale::F16Constant),
                      fused_case(false, KvCacheCase::Mask::Input, KvCacheCase::Scale::ConvertLikeConstant),
                      // a scale computed at run time can't be checked for zero: kept
                      fused_case(false, KvCacheCase::Mask::Input, KvCacheCase::Scale::Input),
                      // the scalar zero mask is not checked by the node when the SDPA is causal
                      fused_case(true, KvCacheCase::Mask::None, KvCacheCase::Scale::Constant),
                      // dynamic head counts (no repeat) are not checked at compile time
                      fused_case(false, KvCacheCase::Mask::None, KvCacheCase::Scale::None, -1),
                      // [B, L, H, D] caches with Transposes: the head axis is permute_axes[1]
                      transposed_case(2, 2)));

// Cases the fused node can't execute: the fusion leaves the SDPA to the generic decomposition.
class StateConcatSDPANotFusedTest : public TransformationTestsF, public WithParamInterface<KvCacheCase> {};

TEST_P(StateConcatSDPANotFusedTest, NotFused) {
    SKIP_ON_ANDROID_X64();
    model = make_gqa_kv_cache_model(GetParam(), false);
    manager.register_pass<StatefulSDPAFusion>();
}

namespace {
KvCacheCase not_fused_case(int64_t rank,
                           int64_t k_heads,
                           int64_t v_heads,
                           KvCacheCase::Mask mask = KvCacheCase::Mask::None,
                           KvCacheCase::Scale scale = KvCacheCase::Scale::None,
                           float scale_value = 0.125F) {
    KvCacheCase c;
    c.rank = rank;
    c.k_heads = k_heads;
    c.v_heads = v_heads;
    c.mask = mask;
    c.scale = scale;
    c.scale_value = scale_value;
    return c;
}
}  // namespace

INSTANTIATE_TEST_SUITE_P(
    TransformationTests,
    StateConcatSDPANotFusedTest,
    ::testing::Values(
        // rank-5 query: grouped and plain
        not_fused_case(5, 2, 2),
        not_fused_case(5, 8, 8),
        // key and value caches with different head counts
        not_fused_case(4, 2, 4),
        // explicit zero scale, also behind a decompression Convert or a ConvertLike
        not_fused_case(4, 2, 2, KvCacheCase::Mask::Input, KvCacheCase::Scale::Constant, 0.0F),
        not_fused_case(4, 2, 2, KvCacheCase::Mask::Input, KvCacheCase::Scale::F16Constant, 0.0F),
        not_fused_case(4, 2, 2, KvCacheCase::Mask::Input, KvCacheCase::Scale::ConvertLikeConstant, 0.0F),
        // scalar zero mask added for a scale without a mask, not causal
        not_fused_case(4, 2, 2, KvCacheCase::Mask::None, KvCacheCase::Scale::Constant),
        // [B, L, H, D] caches with Transposes and different key/value head counts
        transposed_case(2, 4)));

#undef SKIP_ON_ANDROID_X64
