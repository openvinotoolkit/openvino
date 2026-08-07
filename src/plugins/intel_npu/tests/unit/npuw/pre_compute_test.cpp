// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "partitioning/patterns/pre_compute.hpp"

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "openvino/core/except.hpp"
#include "openvino/op/ops.hpp"
#include "orc.hpp"

namespace {

std::shared_ptr<ov::Model> make_longrope_v5_model(const std::vector<float>& short_factor_values,
                                                  const std::vector<float>& long_factor_values,
                                                  const std::vector<float>& multiply_values,
                                                  const std::vector<float>& power_values) {
    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 2});
    auto position_ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{2, 1});

    auto short_factor =
        ov::op::v0::Constant::create(ov::element::f32, ov::Shape{short_factor_values.size()}, short_factor_values);
    auto long_factor =
        ov::op::v0::Constant::create(ov::element::f32, ov::Shape{long_factor_values.size()}, long_factor_values);
    auto multiply_const =
        ov::op::v0::Constant::create(ov::element::f32, ov::Shape{multiply_values.size()}, multiply_values);
    auto power_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{power_values.size()}, power_values);

    auto reduce_axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {0, 1});
    auto red_max = std::make_shared<ov::op::v1::ReduceMax>(position_ids, reduce_axes, false);
    auto one_i32 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {1});
    auto add = std::make_shared<ov::op::v1::Add>(red_max, one_i32);
    auto max_pos = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {4});
    auto greater = std::make_shared<ov::op::v1::Greater>(add, max_pos);

    auto select = std::make_shared<ov::op::v1::Select>(greater, long_factor, short_factor);
    auto multiply = std::make_shared<ov::op::v1::Multiply>(select, multiply_const);
    auto power = std::make_shared<ov::op::v1::Power>(multiply, power_const);

    auto unsqueeze_axis0 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
    auto unsq0 = std::make_shared<ov::op::v0::Unsqueeze>(power, unsqueeze_axis0);
    auto unsq1 = std::make_shared<ov::op::v0::Unsqueeze>(unsq0, unsqueeze_axis0);

    auto shape_of = std::make_shared<ov::op::v3::ShapeOf>(data);
    auto gather_idx0 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
    auto axis0 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
    auto gather = std::make_shared<ov::op::v8::Gather>(shape_of, gather_idx0, axis0);
    auto seq_len = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {4});
    auto rotary_dims = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});
    auto concat_1 = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{gather, seq_len, rotary_dims}, 0);

    auto broadcast = std::make_shared<ov::op::v3::Broadcast>(unsq1, concat_1);
    auto pos_unsq = std::make_shared<ov::op::v0::Unsqueeze>(position_ids, unsqueeze_axis0);
    auto pos_fp32 = std::make_shared<ov::op::v0::Convert>(pos_unsq, ov::element::f32);
    auto matmul = std::make_shared<ov::op::v0::MatMul>(broadcast, pos_fp32);

    auto transpose_order = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {0, 2, 1});
    auto transpose = std::make_shared<ov::op::v1::Transpose>(matmul, transpose_order);
    auto zeros = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 1, 4}, {0.0f, 0.0f, 0.0f, 0.0f});
    auto concat_2 = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{transpose, zeros}, 1);

    auto sin = std::make_shared<ov::op::v0::Sin>(concat_2);
    auto cos = std::make_shared<ov::op::v0::Cos>(concat_2);

    sin->set_friendly_name("sin_out");
    cos->set_friendly_name("cos_out");

    auto sin_res = std::make_shared<ov::op::v0::Result>(sin);
    auto cos_res = std::make_shared<ov::op::v0::Result>(cos);
    return std::make_shared<ov::Model>(ov::ResultVector{sin_res, cos_res},
                                       ov::ParameterVector{data, position_ids},
                                       "longrope_v5_test_model");
}

// Builds a minimal RoPE model matching RopePatternLLama2.
// When with_concat2=true (LLama2 style): Transpose → Concat_2 → Sin/Cos, duplicate_freqs=true.
// When with_concat2=false (GPT style):   Transpose → Sin/Cos directly, duplicate_freqs=false.
std::shared_ptr<ov::Model> make_rope_model(bool with_concat2) {
    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 4});
    auto position_ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1, 4});

    // inv_freq: constant [1, half_dim=2, 1]
    auto inv_freq = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 2, 1}, {0.5f, 0.1f});

    // ShapeOf → Gather(batch dim) → Concat_1 (broadcast target shape)
    auto shape_of = std::make_shared<ov::op::v3::ShapeOf>(data);
    auto gather_idx = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
    auto gather_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
    auto gather = std::make_shared<ov::op::v8::Gather>(shape_of, gather_idx, gather_axis);
    auto ndims_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});
    auto one_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    auto concat_1 = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{gather, ndims_const, one_const}, 0);

    // Broadcast inv_freq to [1,2,1], MatMul with position_ids → Transpose
    auto broadcast = std::make_shared<ov::op::v3::Broadcast>(inv_freq, concat_1);
    auto unsq_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(position_ids, unsq_axis);
    auto convert = std::make_shared<ov::op::v0::Convert>(unsqueeze, ov::element::f32);
    auto matmul = std::make_shared<ov::op::v0::MatMul>(broadcast, convert);  // [1,2,4]
    auto perm = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{0, 2, 1});
    auto transpose = std::make_shared<ov::op::v1::Transpose>(matmul, perm);  // [1,4,2]

    // Sin/Cos either via Concat_2 (LLama2) or directly (GPT)
    ov::Output<ov::Node> sin_cos_input = transpose->output(0);
    if (with_concat2) {
        auto zeros = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 4, 2}, std::vector<float>(8, 0.f));
        sin_cos_input = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{transpose, zeros}, -1)->output(0);
    }

    auto sin = std::make_shared<ov::op::v0::Sin>(sin_cos_input);
    auto cos = std::make_shared<ov::op::v0::Cos>(sin_cos_input);

    return std::make_shared<ov::Model>(
        ov::ResultVector{std::make_shared<ov::op::v0::Result>(sin), std::make_shared<ov::op::v0::Result>(cos)},
        ov::ParameterVector{data, position_ids},
        with_concat2 ? "llama2_rope_test_model" : "gpt_rope_test_model");
}

// Builds a LongRoPE-v5 model that ALSO contains a Phi-style partial-rotary K
// embedding whose result is (a) written out as present.0.key and (b) concatenated
// with a past_key_values.0.key Parameter for attention - i.e. exactly the shape
// CacheRawKeyPattern (NPUW_LLM_LONGROPE_UNROTATED_KV) looks for.
//
// Layout: head_dim = 6, rotary_ndims = 4 (two halves of 2), passthrough = 2,
// past length 7 + 1 current token = full K context of 8.
std::shared_ptr<ov::Model> make_longrope_v5_model_with_kv() {
    auto model = make_longrope_v5_model({1.0f, 2.0f}, {4.0f, 5.0f}, {0.5f, 1.0f}, {2.0f});

    auto raw_k = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 1, 1, 6});
    raw_k->set_friendly_name("raw_k");
    auto past_k = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 1, 7, 6});
    past_k->set_friendly_name("past_key_values.0.key");
    // The K-side cos/sin operands are matched as any_input by the pattern, so plain
    // Parameters keep this test independent of the (separately covered) Q-side rewrite.
    auto cos_in = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 1, 1, 4});
    auto sin_in = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 1, 1, 4});

    auto slice = [](const ov::Output<ov::Node>& data, int64_t begin, int64_t end) {
        auto start = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {begin});
        auto stop = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {end});
        auto step = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
        auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {3});
        return std::make_shared<ov::op::v8::Slice>(data, start, stop, step, axes);
    };

    auto rotary_part = slice(raw_k, 0, 4);
    auto passthrough_part = slice(raw_k, 4, 6);
    auto first_half = slice(rotary_part, 0, 2);
    auto second_half = slice(rotary_part, 2, 4);

    auto minus_one = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {-1.0f});
    auto neg = std::make_shared<ov::op::v1::Multiply>(second_half, minus_one);
    auto rotate_half = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{neg, first_half}, -1);
    auto mul_cos = std::make_shared<ov::op::v1::Multiply>(rotary_part, cos_in);
    auto mul_sin = std::make_shared<ov::op::v1::Multiply>(rotate_half, sin_in);
    auto add = std::make_shared<ov::op::v1::Add>(mul_cos, mul_sin);
    auto k_embed = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{add, passthrough_part}, -1);

    // (a) persisted KV cache output
    auto present_cvt = std::make_shared<ov::op::v0::Convert>(k_embed, ov::element::f16);
    auto present_res = std::make_shared<ov::op::v0::Result>(present_cvt);
    present_res->set_friendly_name("present.0.key");

    // (b) past + current K used by attention
    auto past_cvt = std::make_shared<ov::op::v0::Convert>(past_k, ov::element::f32);
    auto kv_concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{past_cvt, k_embed}, 2);
    auto attn_res = std::make_shared<ov::op::v0::Result>(kv_concat);
    attn_res->set_friendly_name("attention_k");

    model->add_parameters({raw_k, past_k, cos_in, sin_in});
    model->add_results({present_res, attn_res});
    model->validate_nodes_and_infer_types();
    return model;
}

std::shared_ptr<ov::op::v0::Parameter> find_parameter(const std::shared_ptr<ov::Model>& model,
                                                      const std::string& name) {
    for (const auto& param : model->get_parameters()) {
        if (param->get_friendly_name() == name) {
            return param;
        }
    }
    return nullptr;
}

size_t count_parameters(const std::shared_ptr<ov::Model>& model, const std::string& name) {
    const auto& params = model->get_parameters();
    return static_cast<size_t>(std::count_if(params.begin(), params.end(), [&name](const auto& param) {
        return param->get_friendly_name() == name;
    }));
}

bool has_input_name(const std::shared_ptr<ov::Model>& model, const std::string& name) {
    const auto inputs = model->inputs();
    return std::any_of(inputs.begin(), inputs.end(), [&name](const auto& input) {
        const auto& names = input.get_names();
        return std::any_of(names.begin(), names.end(), [&name](const auto& candidate) {
            return candidate == name;
        });
    });
}

TEST(PreComputeTest, RopeCacheTransformsLongRopeV5Pattern) {
    auto model = make_longrope_v5_model({1.0f, 2.0f}, {4.0f, 5.0f}, {0.5f, 1.0f}, {2.0f});

    ov::npuw::patterns::pre_compute::RopeCache pass(/*max_prompt_len=*/16, "longrope_input");
    ASSERT_NO_THROW(pass.run_on_model(model));

    const auto& ops = model->get_ops();
    const auto sin_count = std::count_if(ops.begin(), ops.end(), [](const auto& op) {
        return ov::is_type<ov::op::v0::Sin>(op);
    });
    const auto cos_count = std::count_if(ops.begin(), ops.end(), [](const auto& op) {
        return ov::is_type<ov::op::v0::Cos>(op);
    });

    EXPECT_EQ(sin_count, 0);
    EXPECT_EQ(cos_count, 0);
    EXPECT_TRUE(has_input_name(model, "longrope_input"));
}

TEST(PreComputeTest, RopeCacheThrowsOnMismatchedFactorSizesInLongRopeV5) {
    // multiply has scalar shape {1}: graph is valid by broadcast, but calculate_freq requires exact size match.
    auto model = make_longrope_v5_model({1.0f, 2.0f}, {4.0f, 5.0f}, {1.0f}, {1.0f});
    ov::npuw::patterns::pre_compute::RopeCache pass(/*max_prompt_len=*/16, "longrope_input");

    EXPECT_THROW(pass.run_on_model(model), ov::AssertFailure);
}

TEST(PreComputeTest, RopeCacheThrowsOnNonScalarPowerInLongRopeV5) {
    auto model = make_longrope_v5_model({1.0f, 2.0f}, {4.0f, 5.0f}, {1.0f, 2.0f}, {1.0f, 2.0f});
    ov::npuw::patterns::pre_compute::RopeCache pass(/*max_prompt_len=*/16, "longrope_input");

    EXPECT_THROW(pass.run_on_model(model), ov::AssertFailure);
}

// Verifies that the merged RopePatternLLama2 correctly detects and removes the
// LLama2-style sin/cos subgraph (with Concat_2 present, duplicate_freqs=true).
TEST(PreComputeTest, RopeCacheTransformsLLama2Pattern) {
    auto model = make_rope_model(/*with_concat2=*/true);

    ov::npuw::patterns::pre_compute::RopeCache pass(/*max_prompt_len=*/16, /*longrope_input_name=*/{});
    ASSERT_NO_THROW(pass.run_on_model(model));

    const auto& ops = model->get_ops();
    const auto sin_count = std::count_if(ops.begin(), ops.end(), [](const auto& op) {
        return ov::is_type<ov::op::v0::Sin>(op);
    });
    const auto cos_count = std::count_if(ops.begin(), ops.end(), [](const auto& op) {
        return ov::is_type<ov::op::v0::Cos>(op);
    });

    EXPECT_EQ(sin_count, 0) << "Sin should be replaced by Gather from the duplicated LUT";
    EXPECT_EQ(cos_count, 0) << "Cos should be replaced by Gather from the duplicated LUT";
}

// Verifies that the merged RopePatternLLama2 correctly detects and removes the
// GPT-style sin/cos subgraph (Concat_2 absent, duplicate_freqs=false).
TEST(PreComputeTest, RopeCacheTransformsGPTPattern) {
    auto model = make_rope_model(/*with_concat2=*/false);

    ov::npuw::patterns::pre_compute::RopeCache pass(/*max_prompt_len=*/16, /*longrope_input_name=*/{});
    ASSERT_NO_THROW(pass.run_on_model(model));

    const auto& ops = model->get_ops();
    const auto sin_count = std::count_if(ops.begin(), ops.end(), [](const auto& op) {
        return ov::is_type<ov::op::v0::Sin>(op);
    });
    const auto cos_count = std::count_if(ops.begin(), ops.end(), [](const auto& op) {
        return ov::is_type<ov::op::v0::Cos>(op);
    });

    EXPECT_EQ(sin_count, 0) << "Sin should be replaced by Gather from the non-duplicated LUT";
    EXPECT_EQ(cos_count, 0) << "Cos should be replaced by Gather from the non-duplicated LUT";
}

// NPUW_LLM_LONGROPE_UNROTATED_KV off (the default) must leave the K path untouched.
TEST(PreComputeTest, RopeCacheKeepsRotatedKeyByDefault) {
    auto model = make_longrope_v5_model_with_kv();

    ov::npuw::patterns::pre_compute::RopeCache pass(/*max_prompt_len=*/8, "longrope_input");
    ASSERT_NO_THROW(pass.run_on_model(model));

    EXPECT_FALSE(has_input_name(model, "npuw_lr_full_cos"));
    EXPECT_FALSE(has_input_name(model, "npuw_lr_full_sin"));
    EXPECT_FALSE(pass.host_lut().is_valid());
}

TEST(PreComputeTest, RopeCacheCachesRawKeyAndRotatesAtAttention) {
    auto model = make_longrope_v5_model_with_kv();

    ov::npuw::patterns::pre_compute::RopeCache pass(/*max_prompt_len=*/8,
                                                    "longrope_input",
                                                    /*cache_raw_key_at_attention=*/true);
    ASSERT_NO_THROW(pass.run_on_model(model));

    // The two LUT Parameters are added exactly once, as f16 [1, 1, max_len, head_dim].
    ASSERT_TRUE(has_input_name(model, "npuw_lr_full_cos"));
    ASSERT_TRUE(has_input_name(model, "npuw_lr_full_sin"));
    EXPECT_EQ(count_parameters(model, "npuw_lr_full_cos"), 1u);
    EXPECT_EQ(count_parameters(model, "npuw_lr_full_sin"), 1u);
    for (const auto& name : {"npuw_lr_full_cos", "npuw_lr_full_sin"}) {
        auto param = find_parameter(model, name);
        ASSERT_NE(param, nullptr);
        EXPECT_EQ(param->get_element_type(), ov::element::f16);
        EXPECT_EQ(param->get_shape(), (ov::Shape{1, 1, 8, 6}));
    }

    // present.0.key now stores the RAW key: Result <- Convert <- raw_k Parameter.
    std::shared_ptr<ov::Node> present_src;
    for (const auto& result : model->get_results()) {
        if (result->get_friendly_name() == "present.0.key") {
            present_src = result->input_value(0).get_node_shared_ptr();
        }
    }
    ASSERT_NE(present_src, nullptr);
    ASSERT_TRUE(ov::is_type<ov::op::v0::Convert>(present_src));
    EXPECT_EQ(present_src->input_value(0).get_node()->get_friendly_name(), "raw_k");

    // The attention consumer is now fed by the rotate-at-attention chain
    // Add(Multiply(raw_full, cos), Multiply(rotate_half(raw_full), sin)) instead of
    // the raw past+present Concat directly, and both LUT Parameters reach it through
    // a Convert to the K element type.
    std::shared_ptr<ov::Node> attn_src;
    for (const auto& result : model->get_results()) {
        if (result->get_friendly_name() == "attention_k") {
            attn_src = result->input_value(0).get_node_shared_ptr();
        }
    }
    ASSERT_NE(attn_src, nullptr);
    ASSERT_TRUE(ov::is_type<ov::op::v1::Add>(attn_src));
    auto mul_cos_full = attn_src->input_value(0).get_node_shared_ptr();
    auto mul_sin_full = attn_src->input_value(1).get_node_shared_ptr();
    ASSERT_TRUE(ov::is_type<ov::op::v1::Multiply>(mul_cos_full));
    ASSERT_TRUE(ov::is_type<ov::op::v1::Multiply>(mul_sin_full));
    ASSERT_TRUE(ov::is_type<ov::op::v0::Concat>(mul_cos_full->input_value(0).get_node_shared_ptr()));
    for (const auto& mul : {mul_cos_full, mul_sin_full}) {
        auto lut_cvt = mul->input_value(1).get_node_shared_ptr();
        ASSERT_TRUE(ov::is_type<ov::op::v0::Convert>(lut_cvt));
        EXPECT_EQ(lut_cvt->get_output_element_type(0), ov::element::f32);
        EXPECT_TRUE(ov::is_type<ov::op::v0::Parameter>(lut_cvt->input_value(0).get_node()));
    }

    const auto& lut = pass.host_lut();
    ASSERT_TRUE(lut.is_valid());
    EXPECT_EQ(lut.max_len, 8u);
    EXPECT_EQ(lut.rotary_ndims, 4u);
    EXPECT_EQ(lut.head_dim, 6u);
    EXPECT_EQ(lut.cos_short.get_element_type(), ov::element::f16);
    EXPECT_EQ(lut.cos_short.get_shape(), (ov::Shape{1, 8, 6}));

    // Single source of truth: Q reads a tail slice of the SAME npuw_lr_full_* inputs,
    // so the transformed model carries no cos/sin table of its own - no f16 cache
    // Constant, no short/long Select, and no npuw_longrope_input scalar.
    for (const auto& op : model->get_ops()) {
        if (auto c = ov::as_type_ptr<ov::op::v0::Constant>(op)) {
            EXPECT_NE(c->get_element_type(), ov::element::f16)
                << "a cos/sin cache Constant was emitted next to the host-fed LUT";
        }
        EXPECT_FALSE(ov::is_type<ov::op::v1::Select>(op)) << "the short/long Select should be gone";
    }
    EXPECT_FALSE(has_input_name(model, "longrope_input"));

    // ... and the Q-side Sin/Cos are now fed from npuw_lr_full_cos/sin.
    for (const auto& result : model->get_results()) {
        const auto& rname = result->get_friendly_name();
        if (rname != "present.0.key" && rname != "attention_k") {
            auto node = result->input_value(0).get_node_shared_ptr();
            size_t hops = 0;
            while (node->get_input_size() > 0 && !ov::is_type<ov::op::v0::Parameter>(node) && hops++ < 8) {
                node = node->input_value(0).get_node_shared_ptr();
            }
            EXPECT_THAT(node->get_friendly_name(), ::testing::StartsWith("npuw_lr_full_"));
        }
    }
}

// The LUT metadata has to survive blob export/import, otherwise an imported model
// would run its unconditional raw-K rotation against uninitialized cos/sin inputs.
TEST(PreComputeTest, LongRopeHostLutSerializationRoundTrip) {
    auto model = make_longrope_v5_model_with_kv();
    ov::npuw::patterns::pre_compute::RopeCache pass(/*max_prompt_len=*/8,
                                                    "longrope_input",
                                                    /*cache_raw_key_at_attention=*/true);
    ASSERT_NO_THROW(pass.run_on_model(model));
    const auto& lut = pass.host_lut();
    ASSERT_TRUE(lut.is_valid());

    std::stringstream blob;
    {
        auto writer = ov::npuw::orc::Stream::writer(blob);
        writer & const_cast<ov::npuw::patterns::pre_compute::LongRopeHostLut&>(lut);
    }

    ov::npuw::patterns::pre_compute::LongRopeHostLut restored;
    {
        auto reader = ov::npuw::orc::Stream::reader(blob);
        reader & restored;
    }

    ASSERT_TRUE(restored.is_valid());
    EXPECT_EQ(restored.max_len, lut.max_len);
    EXPECT_EQ(restored.rotary_ndims, lut.rotary_ndims);
    EXPECT_EQ(restored.head_dim, lut.head_dim);

    const std::vector<std::pair<ov::Tensor, ov::Tensor>> pairs{{lut.cos_short, restored.cos_short},
                                                               {lut.sin_short, restored.sin_short},
                                                               {lut.cos_long, restored.cos_long},
                                                               {lut.sin_long, restored.sin_long}};
    for (const auto& [original, imported] : pairs) {
        ASSERT_EQ(imported.get_shape(), original.get_shape());
        ASSERT_EQ(imported.get_element_type(), ov::element::f16);
        EXPECT_EQ(std::memcmp(imported.data(), original.data(), original.get_byte_size()), 0)
            << "rebuilt table differs from the one computed at compile time";
    }
}

}  // namespace
