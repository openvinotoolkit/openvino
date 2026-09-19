// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Unit tests for the static PA variant derivation and the contract it
// requires (ov::npuw::PACompiledModel statics). A synthetic model with the
// CB pipeline's PA contract stands in for a real export: token streams, the
// shared block table, one PagedAttention op per layer over a paged KV cache,
// and the sampled-token gather in front of the LM head.

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "pa_compiled_model.hpp"

namespace {

constexpr int64_t kHidden = 8;
constexpr int64_t kVocab = 10;

std::shared_ptr<ov::op::v0::Parameter> param(const std::string& name,
                                             const ov::element::Type& type,
                                             const ov::PartialShape& shape) {
    auto p = std::make_shared<ov::op::v0::Parameter>(type, shape);
    p->set_friendly_name(name);
    p->output(0).set_names({name});
    return p;
}

std::shared_ptr<ov::Node> empty(const ov::element::Type& type) {
    return ov::op::v0::Constant::create(type, ov::Shape{0}, std::vector<float>{});
}

std::shared_ptr<ov::Node> scalar(const ov::element::Type& type, float value) {
    return ov::op::v0::Constant::create(type, ov::Shape{}, std::vector<float>{value});
}

// A PA-contract model with `layers` attention layers, each one PagedAttention
// op fed by a per-layer paged cache pair. Options let the contract tests
// break it in the ways the front-end must reject.
struct ModelOptions {
    int64_t layers = 2;
    bool with_sampled_gather = true;
    bool embeds_input = false;
    int64_t position_ids_rank = 1;  // 3-D position ids are the M-RoPE (VLM) export
};

std::shared_ptr<ov::Model> make_pa_model(const ModelOptions& o = {}) {
    ov::ParameterVector params;
    const auto add = [&](const std::string& name, const ov::element::Type& type, const ov::PartialShape& shape) {
        auto p = param(name, type, shape);
        params.push_back(p);
        return p;
    };
    std::shared_ptr<ov::Node> hidden;
    if (o.embeds_input) {
        hidden = add("inputs_embeds", ov::element::f32, ov::PartialShape{-1, kHidden});
    } else {
        auto input_ids = add("input_ids", ov::element::i64, ov::PartialShape{-1});
        auto table = ov::op::v0::Constant::create(ov::element::f32,
                                                  ov::Shape{static_cast<std::size_t>(kVocab), kHidden},
                                                  std::vector<float>(kVocab * kHidden, 0.5f));
        hidden = std::make_shared<ov::op::v8::Gather>(table, input_ids, scalar(ov::element::i64, 0));
    }
    add("position_ids", ov::element::i64, ov::PartialShape::dynamic(o.position_ids_rank));
    auto past_lens = add("past_lens", ov::element::i32, ov::PartialShape{-1});
    auto subsequence_begins = add("subsequence_begins", ov::element::i32, ov::PartialShape{-1});
    auto block_indices = add("block_indices", ov::element::i32, ov::PartialShape{-1});
    auto block_indices_begins = add("block_indices_begins", ov::element::i32, ov::PartialShape{-1});
    auto max_context_len = add("max_context_len", ov::element::i32, ov::PartialShape{});

    for (int64_t l = 0; l < o.layers; ++l) {
        const auto tag = std::to_string(l);
        auto key_cache = add("key_cache." + tag, ov::element::dynamic, ov::PartialShape::dynamic(4));
        auto value_cache = add("value_cache." + tag, ov::element::dynamic, ov::PartialShape::dynamic(4));
        auto weight = ov::op::v0::Constant::create(ov::element::f32,
                                                   ov::Shape{kHidden, kHidden},
                                                   std::vector<float>(kHidden * kHidden, 0.1f));
        auto q = std::make_shared<ov::op::v0::MatMul>(hidden, weight);
        auto k = std::make_shared<ov::op::v0::MatMul>(hidden, weight);
        auto v = std::make_shared<ov::op::v0::MatMul>(hidden, weight);
        auto pa = std::make_shared<ov::op::PagedAttentionExtension>(
            ov::OutputVector{q,
                             k,
                             v,
                             key_cache,
                             value_cache,
                             past_lens,
                             subsequence_begins,
                             block_indices,
                             block_indices_begins,
                             scalar(ov::element::f32, 0.125f),  // scale
                             scalar(ov::element::i32, 0),       // sliding_window
                             empty(ov::element::f32),           // alibi_slopes
                             max_context_len,
                             empty(ov::element::i32),      // score_aggregation_window
                             empty(ov::element::i32),      // rotated_block_indices
                             empty(ov::element::i32),      // rotation_deltas
                             empty(ov::element::f32),      // rotation_trig_lut
                             empty(ov::element::f32),      // xattention_threshold
                             scalar(ov::element::i32, 0),  // xattention_block_size
                             scalar(ov::element::i32, 0),  // xattention_stride
                             empty(ov::element::f32),      // sinks
                             scalar(ov::element::i32, 0),  // adaptive_rkv_start_size
                             empty(ov::element::i32),      // adaptive_rkv_evictable_sizes
                             empty(ov::element::i32),      // adaptive_rkv_diversity_block_set_indices
                             empty(ov::element::i32),      // ..._begins
                             empty(ov::element::i32),      // token_type_ids
                             empty(ov::element::u8),       // qq_bias
                             empty(ov::element::i32)},     // qq_bias_begins
            /*write_kv_cache=*/true);
        pa->set_friendly_name("pa." + tag);
        hidden = std::make_shared<ov::op::v1::Add>(hidden, pa->output(0));
    }

    // [tokens, hidden] -> [tokens, 1, hidden], gathered by sampled_tokens_indices, then the LM head.
    std::shared_ptr<ov::Node> rows = std::make_shared<ov::op::v0::Unsqueeze>(hidden, scalar(ov::element::i64, 1));
    if (o.with_sampled_gather) {
        auto sampled = add("sampled_tokens_indices", ov::element::i64, ov::PartialShape{-1});
        rows = std::make_shared<ov::op::v8::Gather>(rows, sampled, scalar(ov::element::i64, 0));
    }
    auto lm_head = ov::op::v0::Constant::create(ov::element::f32,
                                                ov::Shape{kHidden, static_cast<std::size_t>(kVocab)},
                                                std::vector<float>(kHidden * kVocab, 0.2f));
    auto logits = std::make_shared<ov::op::v0::MatMul>(rows, lm_head);
    auto result = std::make_shared<ov::op::v0::Result>(logits);
    result->output(0).set_names({"logits"});
    return std::make_shared<ov::Model>(ov::ResultVector{result}, params, "pa_synthetic");
}

using ov::npuw::PACompiledModel;

std::vector<std::shared_ptr<ov::op::PagedAttentionExtension>> pa_ops(const std::shared_ptr<ov::Model>& m) {
    std::vector<std::shared_ptr<ov::op::PagedAttentionExtension>> out;
    for (const auto& node : m->get_ordered_ops()) {
        if (auto pa = ov::as_type_ptr<ov::op::PagedAttentionExtension>(node)) {
            out.push_back(pa);
        }
    }
    return out;
}

TEST(PAStaticVariant, ContractAcceptsThePipelineModel) {
    EXPECT_NO_THROW(PACompiledModel::require_static_contract(make_pa_model()));
}

TEST(PAStaticVariant, ContractRejectsWithTheReason) {
    const auto expect_reject = [](const ModelOptions& o, const std::string& why) {
        try {
            PACompiledModel::require_static_contract(make_pa_model(o));
            FAIL() << "expected rejection: " << why;
        } catch (const ov::Exception& ex) {
            EXPECT_NE(std::string(ex.what()).find(why), std::string::npos) << ex.what();
        }
    };
    ModelOptions embeds;
    embeds.embeds_input = true;
    expect_reject(embeds, "unsupported input 'inputs_embeds'");
    ModelOptions no_gather;
    no_gather.with_sampled_gather = false;
    expect_reject(no_gather, "missing input 'sampled_tokens_indices'");
    ModelOptions mrope;
    mrope.position_ids_rank = 3;
    expect_reject(mrope, "'position_ids' is not a 1-D token stream");
}

TEST(PAStaticVariant, TokenAndSampledDimsAreFixed) {
    const auto variant = PACompiledModel::derive_static_variant(make_pa_model(), 8u, 4u);
    EXPECT_EQ(variant->input("input_ids").get_partial_shape(), ov::PartialShape({8}));
    EXPECT_EQ(variant->input("position_ids").get_partial_shape(), ov::PartialShape({8}));
    EXPECT_EQ(variant->input("sampled_tokens_indices").get_partial_shape(), ov::PartialShape({4}));
    EXPECT_EQ(variant->output("logits").get_partial_shape(), ov::PartialShape({4, 1, kVocab}));
    // The controls and the cache stay the caller's dynamic tensors.
    EXPECT_TRUE(variant->input("subsequence_begins").get_partial_shape().is_dynamic());
    EXPECT_TRUE(variant->input("key_cache.0").get_partial_shape().is_dynamic());
}

TEST(PAStaticVariant, EveryPagedAttentionSitsInAnIsland) {
    const auto variant = PACompiledModel::derive_static_variant(make_pa_model(), 8u, 4u);
    const auto pas = pa_ops(variant);
    ASSERT_EQ(pas.size(), 2u);

    std::shared_ptr<ov::Node> rows, zeros;
    for (const auto& pa : pas) {
        for (std::size_t i = 0; i < 3; ++i) {  // query, key, value: gathered by the row range
            auto gather = ov::as_type_ptr<ov::op::v8::Gather>(pa->get_input_node_shared_ptr(i));
            ASSERT_TRUE(gather) << "input " << i << " of " << pa->get_friendly_name();
            EXPECT_TRUE(pa->get_input_partial_shape(i).is_dynamic());
            auto range = ov::as_type_ptr<ov::op::v4::Range>(gather->get_input_node_shared_ptr(1));
            ASSERT_TRUE(range);
            EXPECT_TRUE(!rows || rows == range) << "one row range for every island";
            rows = range;
        }
        // The island closes with a scatter into zeros of the static geometry.
        const auto consumers = pa->output(0).get_target_inputs();
        ASSERT_EQ(consumers.size(), 1u);
        auto scatter = ov::as_type_ptr<ov::op::v3::ScatterUpdate>(consumers.begin()->get_node()->shared_from_this());
        ASSERT_TRUE(scatter);
        EXPECT_EQ(scatter->get_output_partial_shape(0), ov::PartialShape({8, kHidden}));
        auto data = ov::as_type_ptr<ov::op::v0::Constant>(scatter->get_input_node_shared_ptr(0));
        ASSERT_TRUE(data);
        EXPECT_TRUE(!zeros || zeros == data) << "one zeros constant shared by the islands";
        zeros = data;
        for (const auto value : data->cast_vector<float>()) {
            EXPECT_EQ(value, 0.f);
        }
    }
    // The row range is the token count subsequence_begins accounts for.
    auto stop = ov::as_type_ptr<ov::op::v8::Gather>(rows->get_input_node_shared_ptr(1));
    ASSERT_TRUE(stop);
    EXPECT_EQ(stop->get_input_node_shared_ptr(0), variant->input("subsequence_begins").get_node_shared_ptr());
}

TEST(PAStaticVariant, EverythingOutsideTheIslandsIsStatic) {
    const auto variant = PACompiledModel::derive_static_variant(make_pa_model(), 8u, 4u);
    for (const auto& node : variant->get_ordered_ops()) {
        const bool island = ov::is_type<ov::op::PagedAttentionExtension>(node) ||
                            ov::is_type<ov::op::v8::Gather>(node) || ov::is_type<ov::op::v4::Range>(node) ||
                            ov::is_type<ov::op::v0::Parameter>(node);
        if (island) {
            continue;
        }
        for (const auto& out : node->outputs()) {
            EXPECT_TRUE(out.get_partial_shape().is_static())
                << node->get_friendly_name() << " " << out.get_partial_shape();
        }
    }
}

}  // namespace
