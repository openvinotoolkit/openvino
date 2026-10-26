// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "llm_test_helpers.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/pass/stateful_to_stateless.hpp"
#include "whisper/prepare_whisper_model.hpp"

namespace {

using ov::test::npuw::build_whisper_decoder_test_model;
using ov::test::npuw::WhisperConfig;

bool has_input_name(const std::shared_ptr<ov::Model>& model, const std::string& substr) {
    for (const auto& in : model->inputs()) {
        for (const auto& name : in.get_names()) {
            if (name.find(substr) != std::string::npos) {
                return true;
            }
        }
    }
    return false;
}

bool has_output_name(const std::shared_ptr<ov::Model>& model, const std::string& substr) {
    for (const auto& out : model->outputs()) {
        for (const auto& name : out.get_names()) {
            if (name.find(substr) != std::string::npos) {
                return true;
            }
        }
    }
    return false;
}

// Replicates the query/key/value part of GenAI's WhisperScaledDotProductAttentionDecomposition
// (and NPUW's own copy of it) for a cross-attention SDPA with no mask input - exactly the shape
// build_whisper_decoder_test_model() produces (encoder_attn SDPA has 3 inputs, no explicit mask).
// This is what a model handed to NPUW already looks like once GenAI decomposes cross-attention
// SDPA for NPU too, matching what it already does for CPU/GPU.
std::shared_ptr<ov::Node> decompose_one_cross_attn_sdpa(const std::shared_ptr<ov::op::v13::ScaledDotProductAttention>& sdpa) {
    using namespace ov::op;
    auto query = sdpa->input_value(0);
    auto key = sdpa->input_value(1);
    auto value = sdpa->input_value(2);

    auto q_shape = std::make_shared<v3::ShapeOf>(query, ov::element::i32);
    auto minus_one = v0::Constant::create(ov::element::i32, ov::Shape{}, {-1});
    auto zero_i = v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    auto one_i = v0::Constant::create(ov::element::i32, ov::Shape{}, {1});
    auto head_dim = std::make_shared<v8::Gather>(q_shape, minus_one, zero_i);
    auto head_dim_f = std::make_shared<v1::ConvertLike>(head_dim, query);
    auto sqrt_hd = std::make_shared<v0::Sqrt>(head_dim_f);
    auto one_f = std::make_shared<v1::ConvertLike>(one_i, query);
    auto scale = std::make_shared<v1::Divide>(one_f, sqrt_hd);

    // Q/K/V are always [batch, heads, seq, head_dim] in this test model - swap the last two axes.
    auto perm = v0::Constant::create(ov::element::i32, ov::Shape{4}, {0, 1, 3, 2});
    auto kT = std::make_shared<v1::Transpose>(key, perm);

    auto q_scaled = std::make_shared<v1::Multiply>(query, scale);
    auto scaled_atten = std::make_shared<v0::MatMul>(q_scaled, kT);
    scaled_atten->output(0).add_names({"cross_attention_qk_scaled_scores"});

    auto softmax = std::make_shared<v8::Softmax>(scaled_atten, -1);
    auto result = std::make_shared<v0::MatMul>(softmax, value);
    result->set_friendly_name(sdpa->get_friendly_name());
    return result;
}

// Decomposes every cross-attention (encoder_attn) SDPA node in-place, simulating a model that
// arrives at NPUW already decomposed by GenAI.
void decompose_cross_attention_sdpa_for_test(const std::shared_ptr<ov::Model>& model) {
    for (const auto& op : model->get_ordered_ops()) {
        auto sdpa = ov::as_type_ptr<ov::op::v13::ScaledDotProductAttention>(op);
        if (!sdpa || sdpa->get_friendly_name().find("encoder_attn") == std::string::npos) {
            continue;
        }
        auto new_node = decompose_one_cross_attn_sdpa(sdpa);
        ov::replace_node(sdpa, new_node);
    }
    model->validate_nodes_and_infer_types();
}

std::shared_ptr<ov::Model> stateless_whisper_decoder_model() {
    auto model = build_whisper_decoder_test_model();
    ov::pass::StatefulToStateless().run_on_model(model);
    return model->clone();
}

class PrepareWhisperModelTest : public ::testing::TestWithParam<bool> {};

// Runs PrepareWhisperPrefillModel/PrepareWhisperKVCacheModel on both a model with fused
// cross-attention SDPA (today's shape) and one with cross-attention SDPA already decomposed
// (the shape NPUW must also handle once GenAI decomposes for NPU, per CVS-184242). Both must
// succeed and produce the same encoder-attn KV-cache input/output names.
TEST_P(PrepareWhisperModelTest, PrefillPreparationHandlesFusedAndDecomposedCrossAttention) {
    const bool decompose = GetParam();

    auto model = stateless_whisper_decoder_model();
    if (decompose) {
        decompose_cross_attention_sdpa_for_test(model);
    }

    WhisperConfig cfg;
    ASSERT_TRUE(ov::npuw::util::PrepareWhisperPrefillModel(128,
                                                           static_cast<uint32_t>(cfg.max_source_positions),
                                                           false /*decompose_sdpa*/)
                    .run_on_model(model));

    EXPECT_TRUE(has_input_name(model, "attention_mask"));
    EXPECT_TRUE(has_output_name(model, "present.0.encoder.key"));
    EXPECT_TRUE(has_output_name(model, "present.0.encoder.value"));
    EXPECT_TRUE(has_output_name(model, "present.1.encoder.key"));
    EXPECT_TRUE(has_output_name(model, "present.1.encoder.value"));
}

TEST_P(PrepareWhisperModelTest, KVCachePreparationHandlesFusedAndDecomposedCrossAttention) {
    const bool decompose = GetParam();

    auto model = stateless_whisper_decoder_model();
    if (decompose) {
        decompose_cross_attention_sdpa_for_test(model);
    }

    ASSERT_TRUE(ov::npuw::util::PrepareWhisperKVCacheModel().run_on_model(model));

    EXPECT_TRUE(has_input_name(model, "past_key_values.0.encoder.key"));
    EXPECT_TRUE(has_input_name(model, "past_key_values.0.encoder.value"));
    EXPECT_TRUE(has_input_name(model, "past_key_values.1.encoder.key"));
    EXPECT_TRUE(has_input_name(model, "past_key_values.1.encoder.value"));
}

INSTANTIATE_TEST_SUITE_P(CrossAttentionShape,
                        PrepareWhisperModelTest,
                        ::testing::Bool(),
                        [](const ::testing::TestParamInfo<bool>& info) {
                            return info.param ? "Decomposed" : "Fused";
                        });

}  // namespace
