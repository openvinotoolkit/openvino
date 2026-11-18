// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "embedding/remove_empty_kv_inputs.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/graph_comparator.hpp"
#include "llm_pass_test_fixture.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/group_query_attention.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/shape_of.hpp"

namespace {

using namespace ov::op;

class RemoveEmptyKVInputsPassTest : public ov::test::npuw::LLMPassTestFixture {};

TEST_F(RemoveEmptyKVInputsPassTest, GQAEmptyPastInputsBecomeConstants) {
    auto query = std::make_shared<v0::Parameter>(ov::element::f16, ov::Shape{1, 4, 1, 16});
    auto key = std::make_shared<v0::Parameter>(ov::element::f16, ov::Shape{1, 2, 1, 16});
    auto value = std::make_shared<v0::Parameter>(ov::element::f16, ov::Shape{1, 2, 1, 16});
    auto past_key = std::make_shared<v0::Parameter>(ov::element::f16, ov::Shape{1, 2, 0, 16});
    auto past_value = std::make_shared<v0::Parameter>(ov::element::f16, ov::Shape{1, 2, 0, 16});
    auto seqlens = std::make_shared<v0::Parameter>(ov::element::i32, ov::Shape{1});
    auto total_length = std::make_shared<v0::Parameter>(ov::element::i32, ov::Shape{1});
    const auto make_model = [&](const ov::Output<ov::Node>& past_key_input,
                                const ov::Output<ov::Node>& past_value_input,
                                bool include_past_parameters) {
        auto gqa = std::make_shared<ov::op::internal::GroupQueryAttention>(
            ov::OutputVector{query, key, value, past_key_input, past_value_input, seqlens, total_length},
            4,
            2,
            0.0f,
            false,
            false);
        ov::ParameterVector parameters{query, key, value};
        if (include_past_parameters) {
            parameters.push_back(past_key);
            parameters.push_back(past_value);
        }
        parameters.push_back(seqlens);
        parameters.push_back(total_length);
        return std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<v0::Result>(gqa->output(0)),
                                                            std::make_shared<v0::Result>(gqa->output(1)),
                                                            std::make_shared<v0::Result>(gqa->output(2))},
                                           parameters);
    };
    auto model = make_model(past_key, past_value, true);
    const auto empty_past_key = v0::Constant::create(ov::element::f16, ov::Shape{1, 2, 0, 16}, {});
    const auto empty_past_value = v0::Constant::create(ov::element::f16, ov::Shape{1, 2, 0, 16}, {});
    const auto model_ref = make_model(empty_past_key, empty_past_value, false);

    ov::npuw::RemoveEmptyKVInputs pass;
    EXPECT_TRUE(pass.run_on_model(model));
    auto comparator = FunctionsComparator::with_default();
    comparator.enable(FunctionsComparator::CONST_VALUES);
    comparator.enable(FunctionsComparator::ATTRIBUTES);
    const auto comparison = comparator.compare(model, model_ref);
    EXPECT_TRUE(comparison.valid) << comparison.message;
    EXPECT_FALSE(pass.run_on_model(model));
}

TEST_F(RemoveEmptyKVInputsPassTest, GQAEmptyPastWindowedCacheBecomesConstants) {
    auto query = std::make_shared<v0::Parameter>(ov::element::f16, ov::Shape{1, 4, 1, 16});
    auto key = std::make_shared<v0::Parameter>(ov::element::f16, ov::Shape{1, 2, 1, 16});
    auto value = std::make_shared<v0::Parameter>(ov::element::f16, ov::Shape{1, 2, 1, 16});
    auto past_key = std::make_shared<v0::Parameter>(ov::element::f16, ov::Shape{1, 2, 0, 16});
    auto past_value = std::make_shared<v0::Parameter>(ov::element::f16, ov::Shape{1, 2, 0, 16});
    auto seqlens = std::make_shared<v0::Parameter>(ov::element::i32, ov::Shape{1});
    auto total_length = std::make_shared<v0::Parameter>(ov::element::i32, ov::Shape{1});
    auto gqa = std::make_shared<ov::op::internal::GroupQueryAttention>(
        ov::OutputVector{query, key, value, past_key, past_value, seqlens, total_length},
        4,
        2,
        0.0f,
        false,
        false,
        0,
        ov::op::internal::GroupQueryAttentionQuantType::NONE,
        ov::op::internal::GroupQueryAttentionQuantType::NONE,
        3,
        true);
    auto model = std::make_shared<ov::Model>(
        ov::ResultVector{std::make_shared<v0::Result>(gqa->output(0)),
                         std::make_shared<v0::Result>(gqa->output(1)),
                         std::make_shared<v0::Result>(gqa->output(2))},
        ov::ParameterVector{query, key, value, past_key, past_value, seqlens, total_length});

    ov::npuw::RemoveEmptyKVInputs pass;
    EXPECT_TRUE(pass.run_on_model(model));
    EXPECT_EQ(model->get_parameters().size(), 5u);
    EXPECT_EQ(model->outputs()[1].get_partial_shape(), (ov::PartialShape{1, 2, 1, 16}));
    EXPECT_EQ(model->outputs()[2].get_partial_shape(), (ov::PartialShape{1, 2, 1, 16}));
    EXPECT_TRUE(ov::is_type<v0::Constant>(gqa->input_value(3).get_node()));
    EXPECT_TRUE(ov::is_type<v0::Constant>(gqa->input_value(4).get_node()));
    EXPECT_FALSE(pass.run_on_model(model));
}

// Direct unit test for RemoveEmptyKVInputs: only the minimal past/current KV concat subgraph
// is needed. This avoids relying on LLMCompiledModel orchestration to validate the matcher.
TEST_F(RemoveEmptyKVInputsPassTest, HandlesDownUpProjSubgraph) {
    auto past_k = std::make_shared<v0::Parameter>(ov::element::f8e4m3, ov::Shape{1, 4, 0, 16});
    past_k->set_friendly_name("past_key_values.0.key");
    past_k->output(0).set_names({"past_key_values.0.key"});

    auto current_k = std::make_shared<v0::Parameter>(ov::element::f8e4m3, ov::Shape{1, 4, 1, 16});
    current_k->set_friendly_name("current_key_values.0.key");
    current_k->output(0).set_names({"current_key_values.0.key"});

    auto upconvert = std::make_shared<v0::Convert>(past_k, ov::element::f32);
    auto upscale = v0::Constant::create(ov::element::f32, ov::Shape{}, {1.0f});
    auto upmul = std::make_shared<v1::Multiply>(upconvert, upscale);
    auto downscale = v0::Constant::create(ov::element::f32, ov::Shape{}, {1.0f});
    auto downmul = std::make_shared<v1::Multiply>(upmul, downscale);
    auto downconvert = std::make_shared<v0::Convert>(downmul, ov::element::f8e4m3);

    auto concat = std::make_shared<v0::Concat>(ov::OutputVector{downconvert, current_k}, 2);
    concat->set_friendly_name("past_key_values.0.keypresent.0.key_concat");

    // ShapeOf on the original parameter exercises the replacement-to-constant path.
    auto shapeof = std::make_shared<v3::ShapeOf>(past_k, ov::element::i64);
    auto concat_result = std::make_shared<v0::Result>(concat);
    concat_result->output(0).set_names({"present.0.key"});
    auto shape_result = std::make_shared<v0::Result>(shapeof);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{concat_result, shape_result},
                                             ov::ParameterVector{past_k, current_k},
                                             "remove_empty_kv_inputs_test");

    ov::npuw::RemoveEmptyKVInputs pass;
    EXPECT_TRUE(pass.run_on_model(model));

    EXPECT_EQ(model->get_parameters().size(), 1u);
    EXPECT_EQ(model->get_parameters().front()->get_friendly_name(), "current_key_values.0.key");
    EXPECT_EQ(count_ops<ov::op::v0::Concat>(model), 0u);
    EXPECT_EQ(count_ops<ov::op::v3::ShapeOf>(model), 0u);

    const auto outputs = model->outputs();
    ASSERT_EQ(outputs.size(), 2u);
    EXPECT_EQ(outputs[0].get_partial_shape(), ov::PartialShape({1, 4, 1, 16}));
}

// Test for shared-KV: same empty past_k parameter is consumed by two independent Concat nodes
// (Gemma4-style shared KV-cache across attention layers).  RemoveEmptyKVInputs must:
//   - eliminate both Concat nodes
//   - remove the shared parameter exactly once (no double-remove crash or assertion)
//   - leave both current-KV parameters in the model
//
// Model topology:
//   past_k[1,4,0,16] ──┬── concat1(axis=2) ── result1
//   current_k1[1,4,1,16]─┘
//                       └── concat2(axis=2) ── result2
//   current_k2[1,4,1,16]──┘
TEST_F(RemoveEmptyKVInputsPassTest, SharedParam_TwoConcats_BothEliminated) {
    auto past_k = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 0, 16});
    past_k->set_friendly_name("past_key_values.0.key");
    past_k->output(0).set_names({"past_key_values.0.key"});

    auto current_k1 = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 1, 16});
    current_k1->set_friendly_name("current_key_values.0.key");
    current_k1->output(0).set_names({"current_key_values.0.key"});

    auto current_k2 = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 1, 16});
    current_k2->set_friendly_name("current_key_values.1.key");
    current_k2->output(0).set_names({"current_key_values.1.key"});

    // Both concat nodes share the same empty past_k (Gemma4 shared KV-cache pattern).
    auto concat1 = std::make_shared<v0::Concat>(ov::OutputVector{past_k, current_k1}, 2);
    concat1->set_friendly_name("past_key_values.0.key_concat1");

    auto concat2 = std::make_shared<v0::Concat>(ov::OutputVector{past_k, current_k2}, 2);
    concat2->set_friendly_name("past_key_values.0.key_concat2");

    auto result1 = std::make_shared<v0::Result>(concat1);
    result1->output(0).set_names({"present.0.key"});
    auto result2 = std::make_shared<v0::Result>(concat2);
    result2->output(0).set_names({"present.1.key"});

    auto model = std::make_shared<ov::Model>(ov::ResultVector{result1, result2},
                                             ov::ParameterVector{past_k, current_k1, current_k2},
                                             "shared_param_two_concats_test");

    ov::npuw::RemoveEmptyKVInputs pass;
    // Must not crash even though the same parameter is matched via two separate concat nodes.
    ASSERT_NO_THROW(pass.run_on_model(model));

    // past_k must have been removed; only the two current-KV params remain.
    EXPECT_EQ(model->get_parameters().size(), 2u);
    // Both Concat nodes must be dead (unreachable from model outputs).
    EXPECT_EQ(count_ops<ov::op::v0::Concat>(model), 0u);
    // Each output now carries only the new single-token KV slice.
    const auto outputs = model->outputs();
    ASSERT_EQ(outputs.size(), 2u);
    EXPECT_EQ(outputs[0].get_partial_shape(), ov::PartialShape({1, 4, 1, 16}));
    EXPECT_EQ(outputs[1].get_partial_shape(), ov::PartialShape({1, 4, 1, 16}));
}

// Same as SharedParam_TwoConcats_BothEliminated but with f8e4m3 past_k routed through
// two independent LPT down/up-convert subgraphs before each Concat (the realistic
// quantised-KV path that triggers the LPT branch of the matcher).
TEST_F(RemoveEmptyKVInputsPassTest, SharedParam_TwoConcats_LptSubgraph_BothEliminated) {
    auto past_k = std::make_shared<v0::Parameter>(ov::element::f8e4m3, ov::Shape{1, 4, 0, 16});
    past_k->set_friendly_name("past_key_values.0.key");
    past_k->output(0).set_names({"past_key_values.0.key"});

    auto current_k1 = std::make_shared<v0::Parameter>(ov::element::f8e4m3, ov::Shape{1, 4, 1, 16});
    current_k1->set_friendly_name("current_key_values.0.key");
    current_k1->output(0).set_names({"current_key_values.0.key"});

    auto current_k2 = std::make_shared<v0::Parameter>(ov::element::f8e4m3, ov::Shape{1, 4, 1, 16});
    current_k2->set_friendly_name("current_key_values.1.key");
    current_k2->output(0).set_names({"current_key_values.1.key"});

    // Build a minimal LPT down/up-convert subgraph rooted at `src`.
    auto make_lpt = [](const std::shared_ptr<ov::Node>& src) -> std::shared_ptr<ov::Node> {
        auto upconv = std::make_shared<v0::Convert>(src, ov::element::f32);
        auto upscale = v0::Constant::create(ov::element::f32, ov::Shape{}, {1.0f});
        auto upmul = std::make_shared<v1::Multiply>(upconv, upscale);
        auto downscale = v0::Constant::create(ov::element::f32, ov::Shape{}, {1.0f});
        auto downmul = std::make_shared<v1::Multiply>(upmul, downscale);
        return std::make_shared<v0::Convert>(downmul, ov::element::f8e4m3);
    };

    // Two independent LPT chains both starting from the same past_k.
    auto lpt1 = make_lpt(past_k);
    auto lpt2 = make_lpt(past_k);
    auto concat1 = std::make_shared<v0::Concat>(ov::OutputVector{lpt1, current_k1}, 2);
    concat1->set_friendly_name("past_key_values.0.key_concat1");
    auto concat2 = std::make_shared<v0::Concat>(ov::OutputVector{lpt2, current_k2}, 2);
    concat2->set_friendly_name("past_key_values.0.key_concat2");

    auto result1 = std::make_shared<v0::Result>(concat1);
    result1->output(0).set_names({"present.0.key"});
    auto result2 = std::make_shared<v0::Result>(concat2);
    result2->output(0).set_names({"present.1.key"});

    auto model = std::make_shared<ov::Model>(ov::ResultVector{result1, result2},
                                             ov::ParameterVector{past_k, current_k1, current_k2},
                                             "shared_param_two_concats_lpt_test");

    ov::npuw::RemoveEmptyKVInputs pass;
    ASSERT_NO_THROW(pass.run_on_model(model));

    EXPECT_EQ(model->get_parameters().size(), 2u);
    EXPECT_EQ(count_ops<ov::op::v0::Concat>(model), 0u);
    const auto outputs = model->outputs();
    ASSERT_EQ(outputs.size(), 2u);
    EXPECT_EQ(outputs[0].get_partial_shape(), ov::PartialShape({1, 4, 1, 16}));
    EXPECT_EQ(outputs[1].get_partial_shape(), ov::PartialShape({1, 4, 1, 16}));
}

// --- Tests for the name-based filter (PR #35221 change) ---
// The pass should only remove parameters whose name matches KV-cache naming conventions.
// Linear cache parameters (cache_params.past.conv/ssm) must be preserved even though
// they structurally match the Parameter->Concat pattern.

// Linear cache conv parameter should NOT be removed by the pass.
TEST_F(RemoveEmptyKVInputsPassTest, LinCacheConvParamPreserved) {
    auto past_conv = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{1, 2048, 0});
    past_conv->set_friendly_name("cache_params.past.conv.0");
    past_conv->output(0).set_names({"cache_params.past.conv.0"});

    auto present_conv = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{1, 2048, 3});
    present_conv->set_friendly_name("cache_params.present.0.conv");
    present_conv->output(0).set_names({"cache_params.present.0.conv"});

    auto concat = std::make_shared<v0::Concat>(ov::OutputVector{past_conv, present_conv}, 2);
    auto result = std::make_shared<v0::Result>(concat);

    auto model = std::make_shared<ov::Model>(ov::ResultVector{result},
                                             ov::ParameterVector{past_conv, present_conv},
                                             "lin_cache_conv_preserved_test");

    ov::npuw::RemoveEmptyKVInputs pass;
    EXPECT_FALSE(pass.run_on_model(model)) << "Pass should not match linear cache conv parameters";
    EXPECT_EQ(model->get_parameters().size(), 2u) << "Both parameters should remain";
    EXPECT_EQ(count_ops<ov::op::v0::Concat>(model), 1u) << "Concat should remain";
}

// Linear cache SSM parameter should NOT be removed by the pass.
TEST_F(RemoveEmptyKVInputsPassTest, LinCacheSsmParamPreserved) {
    auto past_ssm = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{1, 16, 0, 128});
    past_ssm->set_friendly_name("cache_params.past.ssm.0");
    past_ssm->output(0).set_names({"cache_params.past.ssm.0"});

    auto present_ssm = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{1, 16, 128, 128});
    present_ssm->set_friendly_name("cache_params.present.0.ssm");
    present_ssm->output(0).set_names({"cache_params.present.0.ssm"});

    auto concat = std::make_shared<v0::Concat>(ov::OutputVector{past_ssm, present_ssm}, 2);
    auto result = std::make_shared<v0::Result>(concat);

    auto model = std::make_shared<ov::Model>(ov::ResultVector{result},
                                             ov::ParameterVector{past_ssm, present_ssm},
                                             "lin_cache_ssm_preserved_test");

    ov::npuw::RemoveEmptyKVInputs pass;
    EXPECT_FALSE(pass.run_on_model(model)) << "Pass should not match linear cache SSM parameters";
    EXPECT_EQ(model->get_parameters().size(), 2u) << "Both parameters should remain";
    EXPECT_EQ(count_ops<ov::op::v0::Concat>(model), 1u) << "Concat should remain";
}

// Plain f32 key+value pair: both should be removed.
// Also verifies 'value' naming (existing tests only use 'key').
TEST_F(RemoveEmptyKVInputsPassTest, PlainF32KeyValuePairBothRemoved) {
    auto past_k = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 0, 16});
    past_k->set_friendly_name("past_key_values.0.key");
    past_k->output(0).set_names({"past_key_values.0.key"});

    auto present_k = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 1, 16});
    present_k->set_friendly_name("present.0.key");
    present_k->output(0).set_names({"present.0.key"});

    auto past_v = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 0, 16});
    past_v->set_friendly_name("past_key_values.0.value");
    past_v->output(0).set_names({"past_key_values.0.value"});

    auto present_v = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 1, 16});
    present_v->set_friendly_name("present.0.value");
    present_v->output(0).set_names({"present.0.value"});

    auto concat_k = std::make_shared<v0::Concat>(ov::OutputVector{past_k, present_k}, 2);
    auto concat_v = std::make_shared<v0::Concat>(ov::OutputVector{past_v, present_v}, 2);

    auto result_k = std::make_shared<v0::Result>(concat_k);
    result_k->output(0).set_names({"present.0.key"});
    auto result_v = std::make_shared<v0::Result>(concat_v);
    result_v->output(0).set_names({"present.0.value"});

    auto model = std::make_shared<ov::Model>(ov::ResultVector{result_k, result_v},
                                             ov::ParameterVector{past_k, present_k, past_v, present_v},
                                             "plain_f32_key_value_pair_test");

    ov::npuw::RemoveEmptyKVInputs pass;
    EXPECT_TRUE(pass.run_on_model(model));

    // Both past_k and past_v removed; current_k and current_v remain
    EXPECT_EQ(model->get_parameters().size(), 2u);
    EXPECT_EQ(count_ops<ov::op::v0::Concat>(model), 0u);

    const auto outputs = model->outputs();
    ASSERT_EQ(outputs.size(), 2u);
    EXPECT_EQ(outputs[0].get_partial_shape(), ov::PartialShape({1, 4, 1, 16}));
    EXPECT_EQ(outputs[1].get_partial_shape(), ov::PartialShape({1, 4, 1, 16}));
}

// Whisper-style "input_restored." naming should be matched and removed.
TEST_F(RemoveEmptyKVInputsPassTest, WhisperRestoredParamRemoved) {
    const std::string var_name = "input_restored.past_key_values.0.decoder.keypresent.0.decoder.key";
    auto past_k = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{1, 8, 0, 64});
    past_k->set_friendly_name(var_name);
    past_k->output(0).set_names({var_name});

    auto present_k = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{1, 8, 1, 64});
    present_k->set_friendly_name("present.0.key");
    present_k->output(0).set_names({"present.0.key"});

    auto concat = std::make_shared<v0::Concat>(ov::OutputVector{past_k, present_k}, 2);
    auto result = std::make_shared<v0::Result>(concat);
    result->output(0).set_names({"output_restored.past_key_values.0.decoder.key"});

    auto model = std::make_shared<ov::Model>(ov::ResultVector{result},
                                             ov::ParameterVector{past_k, present_k},
                                             "whisper_restored_param_test");

    ov::npuw::RemoveEmptyKVInputs pass;
    EXPECT_TRUE(pass.run_on_model(model));

    EXPECT_EQ(model->get_parameters().size(), 1u);
    EXPECT_EQ(model->get_parameters().front()->get_friendly_name(), "present.0.key");
    EXPECT_EQ(count_ops<ov::op::v0::Concat>(model), 0u);
}

// Arbitrary non-KV parameter name: should NOT be removed.
TEST_F(RemoveEmptyKVInputsPassTest, NonKVNamedParamPreserved) {
    auto past_other = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 0, 16});
    past_other->set_friendly_name("some_other_param");
    past_other->output(0).set_names({"some_other_param"});

    auto present_other = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 1, 16});
    present_other->set_friendly_name("present_other");
    present_other->output(0).set_names({"present_other"});

    auto concat = std::make_shared<v0::Concat>(ov::OutputVector{past_other, present_other}, 2);
    auto result = std::make_shared<v0::Result>(concat);

    auto model = std::make_shared<ov::Model>(ov::ResultVector{result},
                                             ov::ParameterVector{past_other, present_other},
                                             "non_kv_named_param_test");

    ov::npuw::RemoveEmptyKVInputs pass;
    EXPECT_FALSE(pass.run_on_model(model)) << "Pass should not match non-KV named parameters";
    EXPECT_EQ(model->get_parameters().size(), 2u) << "Both parameters should remain";
    EXPECT_EQ(count_ops<ov::op::v0::Concat>(model), 1u) << "Concat should remain";
}

// Hybrid cache model: standard KV key+value (should be removed) + linear cache conv/ssm (should be preserved).
TEST_F(RemoveEmptyKVInputsPassTest, HybridCacheKVRemovedLinCachePreserved) {
    // Standard KV cache key – should be removed
    auto past_k = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 0, 16});
    past_k->set_friendly_name("past_key_values.0.key");
    past_k->output(0).set_names({"past_key_values.0.key"});

    auto present_k = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 1, 16});
    present_k->set_friendly_name("present.0.key");
    present_k->output(0).set_names({"present.0.key"});

    auto concat_k = std::make_shared<v0::Concat>(ov::OutputVector{past_k, present_k}, 2);
    auto result_k = std::make_shared<v0::Result>(concat_k);
    result_k->output(0).set_names({"present.0.key"});

    // Standard KV cache value – should be removed
    auto past_v = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 0, 16});
    past_v->set_friendly_name("past_key_values.0.value");
    past_v->output(0).set_names({"past_key_values.0.value"});

    auto present_v = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 1, 16});
    present_v->set_friendly_name("present.0.value");
    present_v->output(0).set_names({"present.0.value"});

    auto concat_v = std::make_shared<v0::Concat>(ov::OutputVector{past_v, present_v}, 2);
    auto result_v = std::make_shared<v0::Result>(concat_v);
    result_v->output(0).set_names({"present.0.value"});

    // Linear cache conv – should be preserved
    auto past_conv = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{1, 2048, 0});
    past_conv->set_friendly_name("cache_params.past.conv.0");
    past_conv->output(0).set_names({"cache_params.past.conv.0"});

    auto present_conv = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{1, 2048, 3});
    present_conv->set_friendly_name("cache_params.present.conv.0");
    present_conv->output(0).set_names({"cache_params.present.conv.0"});

    auto concat_conv = std::make_shared<v0::Concat>(ov::OutputVector{past_conv, present_conv}, 2);
    auto result_conv = std::make_shared<v0::Result>(concat_conv);

    // Linear cache ssm – should be preserved
    auto past_ssm = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{1, 16, 0, 128});
    past_ssm->set_friendly_name("cache_params.past.ssm.0");
    past_ssm->output(0).set_names({"cache_params.past.ssm.0"});

    auto present_ssm = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{1, 16, 128, 128});
    present_ssm->set_friendly_name("cache_params.present.ssm.0");
    present_ssm->output(0).set_names({"cache_params.present.ssm.0"});

    auto concat_ssm = std::make_shared<v0::Concat>(ov::OutputVector{past_ssm, present_ssm}, 2);
    auto result_ssm = std::make_shared<v0::Result>(concat_ssm);

    auto model = std::make_shared<ov::Model>(
        ov::ResultVector{result_k, result_v, result_conv, result_ssm},
        ov::ParameterVector{past_k, present_k, past_v, present_v, past_conv, present_conv, past_ssm, present_ssm},
        "hybrid_kv_and_lincache_test");

    ov::npuw::RemoveEmptyKVInputs pass;
    EXPECT_TRUE(pass.run_on_model(model));

    // past_k and past_v removed; present_k, present_v, conv pair, ssm pair remain
    EXPECT_EQ(model->get_parameters().size(), 6u) << "Only past_k and past_v should be removed";

    // KV Concats eliminated, linear cache Concats remain
    EXPECT_EQ(count_ops<ov::op::v0::Concat>(model), 2u) << "Only KV Concats should be removed";

    // Verify linear cache parameters are still present
    auto params = model->get_parameters();
    auto has_param = [&](const std::string& name) {
        return std::any_of(params.begin(), params.end(), [&](const auto& p) {
            return p->output(0).get_names().count(name) > 0;
        });
    };
    EXPECT_TRUE(has_param("cache_params.past.conv.0")) << "Conv past should be preserved";
    EXPECT_TRUE(has_param("cache_params.present.conv.0")) << "Conv present should be preserved";
    EXPECT_TRUE(has_param("cache_params.past.ssm.0")) << "SSM past should be preserved";
    EXPECT_TRUE(has_param("cache_params.present.ssm.0")) << "SSM present should be preserved";
}

}  // namespace
