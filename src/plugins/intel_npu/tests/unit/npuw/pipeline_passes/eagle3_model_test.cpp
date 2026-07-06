// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Pipeline-level tests for Eagle3 speculative-decoding models produced by the
// test-engine ModelBuilder: the target model's manually-added
// "last_hidden_state" output must survive the LLM sub-pipeline (and never be
// mistaken for the LM head), and the draft model's Eagle3-specific inputs
// ("hidden_states", "eagle_tree_mask") must be resolved to static shapes by
// ReshapeToStatic via Eagle3Extension::get_static_input.

#include <gtest/gtest.h>

#include "llm_pass_test_fixture.hpp"

namespace {

using ov::test::npuw::RecordingFactory;

class Eagle3ModelTest : public ov::test::npuw::LLMPassTestFixture {
protected:
    static ov::AnyMap eagle_props() {
        return {{"NPUW_EAGLE", "YES"}};
    }
};

// Target model: the "last_hidden_state" output (3-layer concat -> fc) must be
// present and static on both prefill and generate sub-models, alongside logits.
TEST_F(Eagle3ModelTest, TargetModelKeepsLastHiddenStateThroughPipeline) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;
    ASSERT_NO_THROW(
        compiled = create_compiled_model(ov::test::npuw::build_eagle3_target_test_model(), eagle_props(), recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& prefill = require_sub_model(recorder, "_prefill");
    const auto& generate = require_sub_model_containing(recorder, "_kv");

    // fc projects the 3x64 concat back to hidden_size=64
    const auto prefill_hidden = find_output(prefill.model, "last_hidden_state");
    ASSERT_TRUE(prefill_hidden.has_value()) << "last_hidden_state output missing on prefill model";
    ASSERT_TRUE(prefill_hidden->get_partial_shape().is_static());
    EXPECT_EQ(prefill_hidden->get_shape(), (ov::Shape{1, 128, 64}));

    const auto generate_hidden = find_output(generate.model, "last_hidden_state");
    ASSERT_TRUE(generate_hidden.has_value()) << "last_hidden_state output missing on generate model";
    ASSERT_TRUE(generate_hidden->get_partial_shape().is_static());
    EXPECT_EQ(generate_hidden->get_shape(), (ov::Shape{1, 1, 64}));

    // The regular LM head path must be untouched: the default 3-model pipeline
    // splits it into a "_lm_head" sub-model producing "logits", while the
    // prefill model exposes the cut point as "output_embeds".
    const auto& lm_head = require_sub_model(recorder, "_lm_head");
    EXPECT_TRUE(find_output(lm_head.model, "logits").has_value()) << "logits output missing on lm_head model";
    EXPECT_TRUE(find_output(prefill.model, ov::npuw::LLMCompiledModel::output_embeds).has_value())
        << "output_embeds output missing on prefill model";
}

// With a shared LM head, the head cut must pick the real lm_head MatMul and
// skip the manually-added last_hidden_state Result — even though that Result
// is also fed by a MatMul (eagle3_hidden_state_fc).
TEST_F(Eagle3ModelTest, TargetSharedHeadCutSkipsManuallyAddedOutput) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;
    auto props = eagle_props();
    props["NPUW_LLM_SHARED_HEAD"] = "YES";
    ASSERT_NO_THROW(compiled =
                        create_compiled_model(ov::test::npuw::build_eagle3_target_test_model(), props, recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& lm_head = require_sub_model(recorder, "_lm_head");
    (void)lm_head;

    const auto& prefill = require_sub_model(recorder, "_prefill");
    EXPECT_TRUE(find_output(prefill.model, ov::npuw::LLMCompiledModel::output_embeds).has_value())
        << "output_embeds missing on prefill model after LM head cut";

    const auto hidden = find_output(prefill.model, "last_hidden_state");
    ASSERT_TRUE(hidden.has_value()) << "last_hidden_state was consumed by the LM head cut";
    ASSERT_TRUE(hidden->get_partial_shape().is_static());
    EXPECT_EQ(hidden->get_shape(), (ov::Shape{1, 128, 64}));
}

// Draft model (real NPU form): 3 captured layers -> "hidden_states" feature dim
// 3*hidden_size, projected by "model.fc". ReshapeToStatic must fix the static
// width and set the seq dim per phase.
TEST_F(Eagle3ModelTest, DraftHiddenStatesStaticWidthWithFc) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;
    ASSERT_NO_THROW(
        compiled = create_compiled_model(ov::test::npuw::build_eagle3_draft_test_model(), eagle_props(), recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& prefill = require_sub_model(recorder, "_prefill");
    const auto& generate = require_sub_model_containing(recorder, "_kv");

    EXPECT_TRUE(all_inputs_static(prefill.model));
    EXPECT_TRUE(all_inputs_static(generate.model));

    // 3 captured layers * hidden_size(64) = 192
    const auto prefill_hidden = input_shape(prefill.model, "hidden_states");
    ASSERT_TRUE(prefill_hidden.has_value()) << "hidden_states input missing or dynamic on prefill model";
    EXPECT_EQ(*prefill_hidden, (ov::Shape{1, 128, 192}));

    const auto generate_hidden = input_shape(generate.model, "hidden_states");
    ASSERT_TRUE(generate_hidden.has_value()) << "hidden_states input missing or dynamic on generate model";
    EXPECT_EQ(*generate_hidden, (ov::Shape{1, 1, 192}));

    // Draft outputs: the midlayer residual tap (hidden_size wide) rides on both
    // sub-models, and draft-vocab logits land in the split-out "_lm_head" model.
    const auto prefill_hidden_out = find_output(prefill.model, "last_hidden_state");
    ASSERT_TRUE(prefill_hidden_out.has_value());
    ASSERT_TRUE(prefill_hidden_out->get_partial_shape().is_static());
    EXPECT_EQ(prefill_hidden_out->get_shape().back(), 64u);
    EXPECT_TRUE(find_output(generate.model, "last_hidden_state").has_value());

    const auto& lm_head = require_sub_model(recorder, "_lm_head");
    const auto logits = find_output(lm_head.model, "logits");
    ASSERT_TRUE(logits.has_value()) << "logits output missing on lm_head model";
    ASSERT_TRUE(logits->get_partial_shape().is_static());
    EXPECT_EQ(logits->get_shape().back(), 128u) << "logits must use the draft vocab size";
}

// Draft model with a dynamic "hidden_states" feature dim (single captured
// layer, no fc): ReshapeToStatic must recover the width from the
// "last_hidden_state" output via Eagle3Extension::get_static_input.
TEST_F(Eagle3ModelTest, DraftHiddenStatesResolvedFromLastHiddenState) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;
    ASSERT_NO_THROW(compiled = create_compiled_model(ov::test::npuw::build_eagle3_draft_dynamic_hidden_test_model(),
                                                     eagle_props(),
                                                     recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& prefill = require_sub_model(recorder, "_prefill");
    const auto& generate = require_sub_model_containing(recorder, "_kv");

    EXPECT_TRUE(all_inputs_static(prefill.model));
    EXPECT_TRUE(all_inputs_static(generate.model));

    // No fc: width resolves to hidden_size(64) via the last_hidden_state fallback.
    const auto prefill_hidden = input_shape(prefill.model, "hidden_states");
    ASSERT_TRUE(prefill_hidden.has_value()) << "hidden_states input missing or dynamic on prefill model";
    EXPECT_EQ(*prefill_hidden, (ov::Shape{1, 128, 64}));

    const auto generate_hidden = input_shape(generate.model, "hidden_states");
    ASSERT_TRUE(generate_hidden.has_value()) << "hidden_states input missing or dynamic on generate model";
    EXPECT_EQ(*generate_hidden, (ov::Shape{1, 1, 64}));
}

// Draft model with the optional topk tree mask: prefill collapses it to
// {1,1,1,1}; generate spans all past KV positions per token.
TEST_F(Eagle3ModelTest, DraftTreeMaskStaticShapes) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;
    ASSERT_NO_THROW(
        compiled = create_compiled_model(ov::test::npuw::build_eagle3_draft_test_model(true), eagle_props(), recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& prefill = require_sub_model(recorder, "_prefill");
    const auto& generate = require_sub_model_containing(recorder, "_kv");

    const auto prefill_mask = input_shape(prefill.model, "eagle_tree_mask");
    ASSERT_TRUE(prefill_mask.has_value()) << "eagle_tree_mask input missing or dynamic on prefill model";
    EXPECT_EQ(*prefill_mask, (ov::Shape{1, 1, 1, 1}));

    // kvcache size = MAX_PROMPT_LEN(128) + MIN_RESPONSE_LEN(64)
    const auto generate_mask = input_shape(generate.model, "eagle_tree_mask");
    ASSERT_TRUE(generate_mask.has_value()) << "eagle_tree_mask input missing or dynamic on generate model";
    EXPECT_EQ(*generate_mask, (ov::Shape{1, 1, 1, 192}));
}

// Raw-export draft form: the "d2t" output GenAI extracts and strips before the
// model reaches NPUW. Builder-level structural check only — no pipeline run.
TEST(Eagle3DraftBuilderTest, D2tOutputMatchesRawExportForm) {
    auto cfg = ov::test::npuw::make_test_model_config<ov::test::npuw::Eagle3DraftConfig>();
    cfg.draft_vocab_size = 128;
    cfg.with_d2t = true;
    ov::test::npuw::ModelBuilder mb;
    const auto model = mb.build_eagle3_draft(cfg);

    std::shared_ptr<ov::op::v0::Constant> d2t;
    for (const auto& out : model->outputs()) {
        if (out.get_names().count("d2t")) {
            d2t = ov::as_type_ptr<ov::op::v0::Constant>(out.get_node_shared_ptr()->get_input_node_shared_ptr(0));
        }
    }
    ASSERT_NE(d2t, nullptr) << "d2t output missing or not fed by a Constant";
    EXPECT_EQ(d2t->get_element_type(), ov::element::i64);
    ASSERT_EQ(d2t->get_shape(), (ov::Shape{cfg.draft_vocab_size}));

    // Offsets must map the draft vocab into the target vocab strictly monotonically.
    const auto offsets = d2t->cast_vector<int64_t>();
    int64_t prev_target = -1;
    for (size_t i = 0; i < offsets.size(); ++i) {
        const int64_t target = static_cast<int64_t>(i) + offsets[i];
        ASSERT_GE(target, 0) << "d2t maps draft id " << i << " below 0";
        ASSERT_LT(target, static_cast<int64_t>(cfg.vocab_size)) << "d2t maps draft id " << i << " past target vocab";
        ASSERT_GT(target, prev_target) << "d2t mapping must be strictly increasing at draft id " << i;
        prev_target = target;
    }
}

}  // namespace
