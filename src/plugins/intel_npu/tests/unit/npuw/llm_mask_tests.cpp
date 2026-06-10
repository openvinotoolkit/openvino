// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "llm_test_helpers.hpp"
#include "model_builder.hpp"
#include "npuw_transformations/patch_sliding_window_mask.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/openvino.hpp"
#include "openvino/pass/stateful_to_stateless.hpp"

using ov::test::npuw::make_sliding_window_mask_gemma4;
using ov::test::npuw::make_sliding_window_mask_phi3;
using ov::test::npuw::make_sliding_window_mask_phi3_legacy;
using ov::test::npuw::ModelBuilder;

namespace {

bool has_param_named(const std::shared_ptr<ov::Model>& model, const std::string& name) {
    for (const auto& param : model->get_parameters()) {
        if (param->get_friendly_name() == name) {
            return true;
        }
    }
    return false;
}

bool has_sdpa_mask_of_type(const std::shared_ptr<ov::Model>& model, ov::element::Type type) {
    for (const auto& op : model->get_ordered_ops()) {
        auto sdpa = ov::as_type_ptr<ov::op::v13::ScaledDotProductAttention>(op);
        if (sdpa && sdpa->input_value(3).get_element_type() == type) {
            return true;
        }
    }
    return false;
}

}  // namespace

TEST(LLMMaskTest, SlidingWindow_AllLayers_Builds) {
    auto model = ov::test::npuw::build_sliding_window_test_model();
    ASSERT_NE(model, nullptr);
    EXPECT_TRUE(has_param_named(model, "attention_mask"));
}

// Boolean causal mask. NPUW lifts bool SDPA masks to float when decomposing
// SDPA (OptimizeValueTensors) and when preparing the Whisper decoder.
TEST(LLMMaskTest, CausalMask_Boolean_Builds) {
    auto cfg = ov::test::npuw::make_test_model_config();
    cfg.boolean_causal_mask = true;

    ModelBuilder mb;
    auto model = mb.build_llm(cfg);
    ASSERT_NE(model, nullptr);
    EXPECT_TRUE(has_sdpa_mask_of_type(model, ov::element::boolean));
}

// Same boolean causal mask in the Whisper decoder, via the shared toggle on
// BaseModelConfig.
TEST(LLMMaskTest, WhisperDecoder_CausalMask_Boolean_Builds) {
    auto cfg = ov::test::npuw::make_test_model_config<ov::test::npuw::WhisperConfig>();
    cfg.boolean_causal_mask = true;

    ModelBuilder mb;
    auto model = mb.build_whisper_decoder(cfg);
    ASSERT_NE(model, nullptr);
    EXPECT_TRUE(has_sdpa_mask_of_type(model, ov::element::boolean));
}

// Gemma-2 layout: sliding and full attention on alternating layers. Gemma-2/3
// exports use the Phi-3 mask shape (Phi3SlidingMaskMatcher covers all three),
// not the Gemma-4 one.
TEST(LLMMaskTest, SlidingWindow_Gemma2_Builds) {
    auto model = ov::test::npuw::build_sliding_window_test_model(512, 1, make_sliding_window_mask_phi3, 4);
    ASSERT_NE(model, nullptr);
}

// Gemma-3 layout: five sliding layers per full one, same Phi-3 mask shape.
// Needs at least 6 layers to hit both branches of the cycle.
TEST(LLMMaskTest, SlidingWindow_Gemma3_Builds) {
    auto model = ov::test::npuw::build_sliding_window_test_model(1024, 5, make_sliding_window_mask_phi3, 12);
    ASSERT_NE(model, nullptr);
}

// The Phi-3 style mask is boolean and feeds SDPA directly, without a
// Select-to-float on the way.
TEST(LLMMaskTest, SlidingWindow_Phi3Boolean_Builds) {
    auto model = ov::test::npuw::build_sliding_window_test_model(512, 0, make_sliding_window_mask_phi3);
    ASSERT_NE(model, nullptr);
    EXPECT_TRUE(has_sdpa_mask_of_type(model, ov::element::boolean));
}

TEST(LLMMaskTest, TokenTypeIds_HasCorrectInputs) {
    auto model = ov::test::npuw::build_token_type_ids_test_model();
    ASSERT_NE(model, nullptr);
    EXPECT_TRUE(has_param_named(model, "token_type_ids"));
    EXPECT_TRUE(has_param_named(model, "inputs_embeds"));
}

// Reshape to static shapes where seq_len != total_seq, as during generation
// with KV cache. Catches broadcast bugs in the token_type_ids mask modifier,
// which has to slice the query part out of the full-length token_type_ids.
TEST(LLMMaskTest, TokenTypeIds_StaticReshape_SeqNeTotalSeq) {
    auto model = ov::test::npuw::build_token_type_ids_test_model();
    ASSERT_NE(model, nullptr);

    ov::pass::StatefulToStateless().run_on_model(model);
    model = model->clone();

    const int64_t seq_len = 1;
    const int64_t past_kv_len = 16;
    const int64_t total_seq = seq_len + past_kv_len;
    std::map<std::string, ov::PartialShape> new_shapes;
    for (const auto& input : model->inputs()) {
        const auto& name = input.get_any_name();
        const auto& pshape = input.get_partial_shape();
        ov::PartialShape new_shape;
        if (name.find("inputs_embeds") != std::string::npos) {
            new_shape = {1, seq_len, pshape[2]};
        } else if (name.find("attention_mask") != std::string::npos ||
                   name.find("token_type_ids") != std::string::npos) {
            new_shape = {1, total_seq};
        } else if (name.find("position_ids") != std::string::npos) {
            new_shape = {1, seq_len};
        } else if (name.find("beam_idx") != std::string::npos) {
            new_shape = {1};
        } else {
            new_shape = pshape;
            new_shape[0] = 1;
            new_shape[2] = past_kv_len;
        }
        new_shapes[name] = new_shape;
    }

    EXPECT_NO_THROW(model->reshape(new_shapes));
}

// Boolean sliding mask combined with the token_type_ids modifier, like a
// Gemma-3 VLM. The modifier has to stay in the bool domain for bool base
// masks, while the full-attention layers keep the float causal mask.
TEST(LLMMaskTest, TokenTypeIds_Phi3BooleanSlidingMask_Builds) {
    auto model = ov::test::npuw::build_token_type_ids_test_model(512, 1, make_sliding_window_mask_phi3);
    ASSERT_NE(model, nullptr);
    EXPECT_TRUE(has_sdpa_mask_of_type(model, ov::element::boolean));
    EXPECT_TRUE(has_sdpa_mask_of_type(model, ov::element::f32));
}

// NPUW's PatchSlidingWindowMask (sliding_window_mask.cpp) registers three
// matchers: Gemma4SlidingMaskMatcher, Phi3SlidingMaskMatcher (Phi-3, Gemma-2,
// Gemma-3) and OldPhi3SlidingMaskMatcher (transformers 4.51). Run the real
// pass over the builder models and check each mask variant gets picked up.

TEST(LLMMaskTest, NpuwSlidingPatch_FiresOn_Phi3Pattern) {
    auto model = ov::test::npuw::build_sliding_window_test_model(512, 0, make_sliding_window_mask_phi3);
    EXPECT_TRUE(ov::npuw::PatchSlidingWindowMask().run_on_model(model));
    EXPECT_NO_THROW(model->validate_nodes_and_infer_types());
}

TEST(LLMMaskTest, NpuwSlidingPatch_FiresOn_Gemma4Pattern) {
    auto model = ov::test::npuw::build_sliding_window_test_model(512, 0, make_sliding_window_mask_gemma4);
    EXPECT_TRUE(ov::npuw::PatchSlidingWindowMask().run_on_model(model));
    EXPECT_NO_THROW(model->validate_nodes_and_infer_types());
}

TEST(LLMMaskTest, NpuwSlidingPatch_FiresOn_LegacyPhi3Pattern) {
    auto model = ov::test::npuw::build_sliding_window_test_model(512, 0, make_sliding_window_mask_phi3_legacy);
    EXPECT_TRUE(ov::npuw::PatchSlidingWindowMask().run_on_model(model));
    EXPECT_NO_THROW(model->validate_nodes_and_infer_types());
}

// The default float SWA mask intentionally matches none of the NPUW patterns.
TEST(LLMMaskTest, NpuwSlidingPatch_IgnoresDefaultFloatMask) {
    auto model = ov::test::npuw::build_sliding_window_test_model(512, 0);
    EXPECT_FALSE(ov::npuw::PatchSlidingWindowMask().run_on_model(model));
}

// Real Gemma exports build two mask subgraphs and share them across all
// layers: one sliding, one full, with the sliding mask on layer 0. Check
// the builder produces the same layout.
TEST(LLMMaskTest, SlidingFullAlternation_MatchesRealGemmaLayout) {
    auto model = ov::test::npuw::build_sliding_window_test_model(512, 1, {}, 4);
    ASSERT_NE(model, nullptr);

    std::map<size_t, const ov::Node*> layer_mask;
    for (const auto& op : model->get_ordered_ops()) {
        auto sdpa = ov::as_type_ptr<ov::op::v13::ScaledDotProductAttention>(op);
        if (!sdpa) {
            continue;
        }
        const auto& name = sdpa->get_friendly_name();
        const std::string marker = "model.layers.";
        auto pos = name.find(marker);
        ASSERT_NE(pos, std::string::npos) << name;
        layer_mask[std::stoul(name.substr(pos + marker.size()))] = sdpa->input_value(3).get_node();
    }
    ASSERT_EQ(layer_mask.size(), 4u);

    // Sliding on even layers, full on odd, one shared mask node per kind.
    EXPECT_EQ(layer_mask[0], layer_mask[2]);
    EXPECT_EQ(layer_mask[1], layer_mask[3]);
    EXPECT_NE(layer_mask[0], layer_mask[1]);
    EXPECT_NE(layer_mask[0]->get_friendly_name().find("model.sw."), std::string::npos);
    EXPECT_EQ(layer_mask[1]->get_friendly_name().find("model.sw."), std::string::npos);
}
