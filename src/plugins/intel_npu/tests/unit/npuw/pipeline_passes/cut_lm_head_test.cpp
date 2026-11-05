// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "npuw_transformations/cut_lm_head.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "llm_compiled_model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/subtract.hpp"

namespace {

using ov::npuw::LLMCompiledModel;

constexpr int64_t kSeq = 8;
constexpr int64_t kHidden = 16;
constexpr int64_t kVocab = 32;

std::shared_ptr<ov::op::v0::Constant> make_weights() {
    return ov::op::v0::Constant::create(ov::element::f32,
                                        ov::Shape{static_cast<size_t>(kHidden), static_cast<size_t>(kVocab)},
                                        std::vector<float>(kHidden * kVocab, 0.0f));
}

// Parameter(hidden) -> MatMul(hidden, weights) -> Result("logits").
std::shared_ptr<ov::Model> build_simple_lm_head_model() {
    auto hidden = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, kSeq, kHidden});
    hidden->set_friendly_name("hidden");
    hidden->output(0).set_names({"hidden"});

    auto matmul = std::make_shared<ov::op::v0::MatMul>(hidden, make_weights());
    auto result = std::make_shared<ov::op::v0::Result>(matmul);
    result->set_friendly_name("logits");
    result->output(0).set_names({LLMCompiledModel::layer_names::logits});

    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{hidden}, "simple_lm_head");
}

// Same, but hidden also feeds a second Result ("last_hidden_state"); mimics the
// OmniThinker case where the pre-head embeddings are already a model output.
std::shared_ptr<ov::Model> build_attached_result_model() {
    auto hidden = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, kSeq, kHidden});
    hidden->set_friendly_name("hidden");
    hidden->output(0).set_names({"hidden"});

    auto hidden_result = std::make_shared<ov::op::v0::Result>(hidden);
    hidden_result->set_friendly_name("last_hidden_state");
    hidden_result->output(0).set_names({"last_hidden_state"});

    auto matmul = std::make_shared<ov::op::v0::MatMul>(hidden, make_weights());
    auto logits_result = std::make_shared<ov::op::v0::Result>(matmul);
    logits_result->set_friendly_name("logits");
    logits_result->output(0).set_names({LLMCompiledModel::layer_names::logits});

    return std::make_shared<ov::Model>(ov::ResultVector{hidden_result, logits_result},
                                       ov::ParameterVector{hidden},
                                       "attached_result_lm_head");
}

std::shared_ptr<ov::op::v0::Result> find_result_by_name(const std::shared_ptr<ov::Model>& model,
                                                        const std::string& name) {
    for (const auto& r : model->get_results()) {
        if (r->output(0).get_names().count(name)) {
            return r;
        }
    }
    return nullptr;
}

}  // namespace

// --- Test 1 -------------------------------------------------------------------
// Basic cut: no Result attached at the MatMul input source. The logits Result is
// repurposed as the output-embeddings Result, and the LM head MatMul is moved
// into a dedicated sub-model.
TEST(CutLMHeadTest, BasicCut) {
    auto model = build_simple_lm_head_model();
    ASSERT_EQ(model->get_results().size(), 1u);

    std::shared_ptr<ov::Model> lm_head_model;
    const bool changed = ov::npuw::CutLMHead(lm_head_model).run_on_model(model);

    EXPECT_TRUE(changed);
    ASSERT_NE(lm_head_model, nullptr);

    // Original model: the logits Result is renamed to output_embeds and now reads
    // directly from the hidden Parameter (the MatMul moved to the LM head sub-model).
    ASSERT_EQ(model->get_results().size(), 1u);
    const auto embeds = find_result_by_name(model, LLMCompiledModel::layer_names::output_embeds);
    ASSERT_NE(embeds, nullptr);
    EXPECT_EQ(embeds->output(0).get_names().count(LLMCompiledModel::layer_names::logits), 0u);
    EXPECT_EQ(embeds->input_value(0).get_node(), model->get_parameters().front().get());

    // LM head sub-model: single Parameter (named output_embeds) feeding the LM head MatMul.
    ASSERT_EQ(lm_head_model->get_parameters().size(), 1u);
    ASSERT_EQ(lm_head_model->get_results().size(), 1u);
    const auto& head_param = lm_head_model->get_parameters().front();
    EXPECT_GT(head_param->output(0).get_names().count(LLMCompiledModel::layer_names::output_embeds), 0u);
}

// --- Test 2 -------------------------------------------------------------------
// Attached-Result case: a Result already exists on the MatMul input source (the
// OmniThinker pre-head embeddings output). The pass repurposes the matched logits
// Result as the output-embeddings Result and keeps the pre-existing Result. To keep
// the two outputs distinct, the embeddings Result reads a pass-through Convert while
// the pre-existing Result keeps reading the shared producer directly.
TEST(CutLMHeadTest, AttachedResultKeepsBothOutputs) {
    auto model = build_attached_result_model();
    ASSERT_EQ(model->get_results().size(), 2u);

    std::shared_ptr<ov::Model> lm_head_model;
    const bool changed = ov::npuw::CutLMHead(lm_head_model).run_on_model(model);

    EXPECT_TRUE(changed);
    ASSERT_NE(lm_head_model, nullptr);

    // Original model: the pre-existing Result stays and the logits Result is repurposed
    // as the output-embeddings Result; logits is gone.
    ASSERT_EQ(model->get_results().size(), 2u);
    const auto hidden_result = find_result_by_name(model, "last_hidden_state");
    const auto embeds_result = find_result_by_name(model, LLMCompiledModel::layer_names::output_embeds);
    ASSERT_NE(hidden_result, nullptr);
    ASSERT_NE(embeds_result, nullptr);
    EXPECT_EQ(find_result_by_name(model, LLMCompiledModel::layer_names::logits), nullptr);

    const auto& hidden_param = model->get_parameters().front();
    // The pre-existing Result keeps reading the shared producer directly.
    EXPECT_EQ(hidden_result->input_value(0).get_node(), hidden_param.get());
    // The embeddings Result reads a distinct pass-through Convert over the same producer.
    const auto embeds_src = embeds_result->input_value(0).get_node_shared_ptr();
    EXPECT_NE(embeds_src.get(), hidden_param.get());
    const auto embeds_convert = ov::as_type_ptr<ov::op::v0::Convert>(embeds_src);
    ASSERT_NE(embeds_convert, nullptr);
    EXPECT_EQ(embeds_convert->input_value(0).get_node(), hidden_param.get());

    // LM head sub-model: single Parameter feeding the LM head MatMul.
    ASSERT_EQ(lm_head_model->get_parameters().size(), 1u);
    ASSERT_EQ(lm_head_model->get_results().size(), 1u);
}

// --- Test 3 -------------------------------------------------------------------
// A model without a logits Result must not be touched; lm_head_model stays null.
TEST(CutLMHeadTest, NoLogitsIsUntouched) {
    auto hidden = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, kSeq, kHidden});
    hidden->set_friendly_name("hidden");
    hidden->output(0).set_names({"hidden"});

    auto matmul = std::make_shared<ov::op::v0::MatMul>(hidden, make_weights());
    auto result = std::make_shared<ov::op::v0::Result>(matmul);
    result->set_friendly_name("not_logits");
    result->output(0).set_names({"not_logits"});

    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{hidden}, "no_logits_model");

    std::shared_ptr<ov::Model> lm_head_model;
    const bool changed = ov::npuw::CutLMHead(lm_head_model).run_on_model(model);

    EXPECT_FALSE(changed);
    EXPECT_EQ(lm_head_model, nullptr);
    ASSERT_EQ(model->get_results().size(), 1u);
    EXPECT_EQ(find_result_by_name(model, "not_logits"), result);
}

class CutLMHeadTiedWeightsTest : public testing::TestWithParam<bool> {};

TEST_P(CutLMHeadTiedWeightsTest, ConstantLiftingDoesNotModifyOtherModel) {
    auto ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1, kSeq});
    auto weights = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{kVocab, kHidden}, {1});
    auto weights_f32 = std::make_shared<ov::op::v0::Convert>(weights, ov::element::f32);
    auto zero_point = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{kVocab, 1}, {1});
    auto zero_point_f32 = std::make_shared<ov::op::v0::Convert>(zero_point, ov::element::f32);
    zero_point_f32->set_friendly_name("tied_zero_point_convert");
    auto subtract = std::make_shared<ov::op::v1::Subtract>(weights_f32, zero_point_f32);
    auto scale = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{kVocab, 1}, {0.5f});
    auto dequantized = std::make_shared<ov::op::v1::Multiply>(subtract, scale);
    auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
    auto embeddings = std::make_shared<ov::op::v8::Gather>(dequantized, ids, axis);
    auto matmul = std::make_shared<ov::op::v0::MatMul>(embeddings, dequantized, false, true);
    auto result = std::make_shared<ov::op::v0::Result>(matmul);
    result->output(0).set_names({LLMCompiledModel::layer_names::logits});
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{ids});

    std::shared_ptr<ov::Model> lm_head_model;
    ASSERT_TRUE(ov::npuw::CutLMHead(lm_head_model).run_on_model(model));
    ASSERT_NE(lm_head_model, nullptr);
    ASSERT_NO_THROW(model->validate_nodes_and_infer_types());
    ASSERT_NO_THROW(lm_head_model->validate_nodes_and_infer_types());

    const auto& modified_model = GetParam() ? lm_head_model : model;
    const auto& other_model = GetParam() ? model : lm_head_model;
    std::shared_ptr<ov::Node> modified_convert;
    std::shared_ptr<ov::Node> other_convert;
    for (const auto& op : modified_model->get_ops()) {
        if (op->get_friendly_name() == "tied_zero_point_convert") {
            modified_convert = op;
        }
    }
    for (const auto& op : other_model->get_ops()) {
        if (op->get_friendly_name() == "tied_zero_point_convert") {
            other_convert = op;
        }
    }
    ASSERT_NE(modified_convert, nullptr);
    ASSERT_NE(other_convert, nullptr);
    auto other_zero_point = ov::as_type_ptr<ov::op::v0::Constant>(other_convert->input_value(0).get_node_shared_ptr());
    ASSERT_NE(other_zero_point, nullptr);
    EXPECT_EQ(other_zero_point->get_data_ptr(), zero_point->get_data_ptr());

    // NPUW function creation lifts constants into registered closure parameters.
    auto lifted = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, zero_point->get_shape());
    modified_convert->input(0).replace_source_output(lifted);
    modified_model->add_parameters({lifted});

    EXPECT_NE(modified_convert, other_convert);
    EXPECT_TRUE(ov::is_type<ov::op::v0::Constant>(other_convert->input_value(0).get_node()));
    EXPECT_NO_THROW(modified_model->validate_nodes_and_infer_types());
    EXPECT_NO_THROW(other_model->validate_nodes_and_infer_types());
    EXPECT_EQ(lm_head_model->get_parameters().size(), GetParam() ? 2u : 1u);
}

INSTANTIATE_TEST_SUITE_P(CutLMHeadTest, CutLMHeadTiedWeightsTest, testing::Bool());
