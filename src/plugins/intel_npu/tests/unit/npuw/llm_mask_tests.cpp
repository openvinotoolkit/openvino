// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "llm_test_helpers.hpp"
#include "model_builder.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/openvino.hpp"
#include "openvino/pass/stateful_to_stateless.hpp"

using ov::test::npuw::LLMConfig;
using ov::test::npuw::make_causal_mask_boolean;
using ov::test::npuw::make_sliding_window_mask_phi3;
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

}  // namespace

TEST(LLMMaskTest, SlidingWindow_AllLayers_Builds) {
    auto model = ov::test::npuw::build_sliding_window_test_model();
    ASSERT_NE(model, nullptr);
    EXPECT_TRUE(has_param_named(model, "attention_mask"));
}

// Boolean causal mask variant — exercises NPUW handlers that lift bool SDPA masks
// to float (optimize_value_tensors.cpp:271, prepare_whisper_model.cpp:140).
TEST(LLMMaskTest, CausalMask_Boolean_Builds) {
    auto cfg = ov::test::npuw::make_test_model_config();
    cfg.causal_mask_fn = make_causal_mask_boolean;

    ModelBuilder mb;
    auto model = mb.build_llm(cfg);
    ASSERT_NE(model, nullptr);

    bool found_bool_sdpa_mask = false;
    for (const auto& op : model->get_ordered_ops()) {
        auto sdpa = ov::as_type_ptr<ov::op::v13::ScaledDotProductAttention>(op);
        if (!sdpa) {
            continue;
        }
        if (sdpa->input_value(3).get_element_type() == ov::element::boolean) {
            found_bool_sdpa_mask = true;
            break;
        }
    }
    EXPECT_TRUE(found_bool_sdpa_mask);
}

// Whisper decoder boolean causal mask — covers the bool path of
// prepare_whisper_model.cpp:140 (Select(mask, 0, -inf) on bool source).
// The shared causal_mask_fn on BaseModelConfig acts as the toggle.
TEST(LLMMaskTest, WhisperDecoder_CausalMask_Boolean_Builds) {
    auto cfg = ov::test::npuw::make_test_model_config<ov::test::npuw::WhisperConfig>();
    cfg.causal_mask_fn = make_causal_mask_boolean;

    ModelBuilder mb;
    auto model = mb.build_whisper_decoder(cfg);
    ASSERT_NE(model, nullptr);

    bool found_bool_sdpa_mask = false;
    for (const auto& op : model->get_ordered_ops()) {
        auto sdpa = ov::as_type_ptr<ov::op::v13::ScaledDotProductAttention>(op);
        if (!sdpa) {
            continue;
        }
        if (sdpa->input_value(3).get_element_type() == ov::element::boolean) {
            found_bool_sdpa_mask = true;
            break;
        }
    }
    EXPECT_TRUE(found_bool_sdpa_mask);
}

TEST(LLMMaskTest, SlidingWindow_Alternating_Builds) {
    auto model = ov::test::npuw::build_sliding_window_test_model(512, true);
    ASSERT_NE(model, nullptr);
}

// Phi-3-style boolean SWA pattern — verifies the model builds and at least one
// SDPA receives a boolean mask (matching Phi3SlidingMaskMatcher's expected shape).
TEST(LLMMaskTest, SlidingWindow_Phi3Boolean_Builds) {
    auto cfg = ov::test::npuw::make_test_model_config();
    cfg.sliding_window_size = 512;
    cfg.alternating_attention = false;
    cfg.sliding_mask_fn = make_sliding_window_mask_phi3;

    ModelBuilder mb;
    auto model = mb.build_llm(cfg);
    ASSERT_NE(model, nullptr);

    bool found_bool_sdpa_mask = false;
    for (const auto& op : model->get_ordered_ops()) {
        auto sdpa = ov::as_type_ptr<ov::op::v13::ScaledDotProductAttention>(op);
        if (!sdpa) {
            continue;
        }
        if (sdpa->input_value(3).get_element_type() == ov::element::boolean) {
            found_bool_sdpa_mask = true;
            break;
        }
    }
    EXPECT_TRUE(found_bool_sdpa_mask);
}

// Phi-3 boolean pattern + alternating layers — even layers boolean SWA, odd float causal.
TEST(LLMMaskTest, SlidingWindow_Phi3Boolean_Alternating_Builds) {
    auto cfg = ov::test::npuw::make_test_model_config();
    cfg.num_layers = 4;
    cfg.sliding_window_size = 512;
    cfg.alternating_attention = true;
    cfg.sliding_mask_fn = make_sliding_window_mask_phi3;

    ModelBuilder mb;
    auto model = mb.build_llm(cfg);
    ASSERT_NE(model, nullptr);
}

TEST(LLMMaskTest, TokenTypeIds_HasCorrectInputs) {
    auto model = ov::test::npuw::build_token_type_ids_test_model();
    ASSERT_NE(model, nullptr);
    EXPECT_TRUE(has_param_named(model, "token_type_ids"));
    EXPECT_TRUE(has_param_named(model, "inputs_embeds"));
}

// KV-cache scenario where seq_len != total_seq — exercises the offset slicing
// in make_vlm_bidirectional_modifier to catch broadcast/shape regressions.
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
