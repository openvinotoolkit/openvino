// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "npuw_transformations/cut_lm_head.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "llm_compiled_model.hpp"
#include "npuw_transformations/insert_vocab_sub128.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/subtract.hpp"
#include "partitioning/patterns/opt.hpp"

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

// Tied u8 vocabulary: Gather and the LM-head MatMul share one dequantization chain.
std::shared_ptr<ov::Model> build_shared_u8_vocabulary_model(const ov::element::Type& precision) {
    auto hidden = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, kSeq, kHidden});
    auto ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{1, kSeq});

    const std::vector<uint8_t> weight_values{0, 127, 128, 255, 1, 42, 253, 254};
    const std::vector<uint8_t> zero_point_values{128, 127, 0, 255};
    std::vector<uint8_t> weight_data(kVocab * kHidden);
    std::vector<uint8_t> zero_point_data(kVocab);
    for (size_t i = 0; i < weight_data.size(); ++i) {
        weight_data[i] = weight_values[i % weight_values.size()];
    }
    for (size_t i = 0; i < zero_point_data.size(); ++i) {
        zero_point_data[i] = zero_point_values[i % zero_point_values.size()];
    }
    auto weights = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{kVocab, kHidden}, weight_data);
    auto weights_convert = std::make_shared<ov::op::v0::Convert>(weights, precision);
    auto zero_points = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{kVocab, 1}, zero_point_data);
    auto zero_points_convert = std::make_shared<ov::op::v0::Convert>(zero_points, precision);
    auto dequantized = std::make_shared<ov::op::v1::Subtract>(weights_convert, zero_points_convert);
    auto scales = ov::op::v0::Constant::create(precision, ov::Shape{kVocab, 1}, {0.5f});
    auto scaled = std::make_shared<ov::op::v1::Multiply>(dequantized, scales);
    auto converted = std::make_shared<ov::op::v0::Convert>(scaled, ov::element::f32);

    auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
    auto gather = std::make_shared<ov::op::v8::Gather>(converted, ids, axis);
    auto embeddings = std::make_shared<ov::op::v0::Result>(gather);
    embeddings->output(0).set_names({"embeddings"});

    auto matmul = std::make_shared<ov::op::v0::MatMul>(hidden, converted, false, true);
    auto logits = std::make_shared<ov::op::v0::Result>(matmul);
    logits->output(0).set_names({LLMCompiledModel::layer_names::logits});

    return std::make_shared<ov::Model>(ov::ResultVector{embeddings, logits}, ov::ParameterVector{hidden, ids});
}

std::size_t count_marked_sub128_shifts(const std::shared_ptr<ov::Model>& model) {
    const auto ops = model->get_ordered_ops();
    return std::count_if(ops.begin(), ops.end(), [](const auto& node) {
        return node->get_rt_info().count(ov::npuw::NPUW_SUB128_SHIFT_RT_INFO) > 0;
    });
}

void expect_equal_f32_tensors(const ov::Tensor& expected, const ov::Tensor& actual) {
    ASSERT_EQ(actual.get_shape(), expected.get_shape());
    for (size_t i = 0; i < expected.get_size(); ++i) {
        EXPECT_EQ(actual.data<float>()[i], expected.data<float>()[i]);
    }
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

class CutLMHeadSharedVocabTest : public testing::TestWithParam<ov::element::Type> {};

TEST_P(CutLMHeadSharedVocabTest, PreservesQuantizedVocabularyAndOutputs) {
    using namespace ov::npuw::patterns::opt;
    auto model = build_shared_u8_vocabulary_model(GetParam());
    const auto preserve = [](const std::shared_ptr<ov::Model>& graph, bool restore_sub128 = false) {
        Context ctx(graph, graph->get_parameters().size());
        std::vector<std::shared_ptr<ov::op::v0::Constant>> constants;
        ov::pass::GraphRewrite rewrite;
        rewrite.add_matcher<PreserveConstDictMatMulAsymm>(std::ref(ctx), std::ref(constants), restore_sub128);
        rewrite.run_on_model(graph);
        return constants;
    };
    const auto original_constants = preserve(model);
    ASSERT_EQ(original_constants.size(), 3u);

    ov::Tensor hidden(ov::element::f32, ov::Shape{1, kSeq, kHidden});
    for (size_t i = 0; i < hidden.get_size(); ++i) {
        hidden.data<float>()[i] = (static_cast<int>(i % 7) - 3) * 0.25f;
    }
    ov::Tensor ids(ov::element::i64, ov::Shape{1, kSeq});
    for (size_t i = 0; i < ids.get_size(); ++i) {
        ids.data<int64_t>()[i] = i;
    }
    ov::TensorVector expected{ov::Tensor(ov::element::f32, ov::Shape{1, kSeq, kHidden}),
                              ov::Tensor(ov::element::f32, ov::Shape{1, kSeq, kVocab})};
    ASSERT_TRUE(model->evaluate(expected, {hidden, ids}));
    ASSERT_TRUE(
        std::any_of(expected[1].data<float>(), expected[1].data<float>() + expected[1].get_size(), [](float value) {
            return value != 0.0f;
        }));

    ASSERT_TRUE(ov::npuw::InsertVocabSub128().run_on_model(model));
    // Normalization must also isolate the MatMul branch when Gather still shares the DQ nodes.
    auto shared_model = model->clone();
    ASSERT_EQ(preserve(shared_model, true).size(), 3u);
    EXPECT_EQ(count_marked_sub128_shifts(shared_model), 2u);
    ov::TensorVector shared_outputs{ov::Tensor(ov::element::f32, ov::Shape{1, kSeq, kHidden}),
                                    ov::Tensor(ov::element::f32, ov::Shape{1, kSeq, kVocab})};
    ASSERT_TRUE(shared_model->evaluate(shared_outputs, {hidden, ids}));
    expect_equal_f32_tensors(expected[0], shared_outputs[0]);
    expect_equal_f32_tensors(expected[1], shared_outputs[1]);

    std::shared_ptr<ov::Model> lm_head_model;
    ASSERT_TRUE(ov::npuw::CutLMHead(lm_head_model).run_on_model(model));
    ASSERT_NE(lm_head_model, nullptr);
    EXPECT_EQ(count_marked_sub128_shifts(model), 2u);
    EXPECT_EQ(count_marked_sub128_shifts(lm_head_model), 2u);

    auto legacy_head = lm_head_model->clone();
    // Detection retains shifts; only the selected modern quantized-tail path restores canonical U8 DQ.
    ASSERT_EQ(preserve(lm_head_model).size(), 3u);
    EXPECT_EQ(count_marked_sub128_shifts(lm_head_model), 2u);
    const auto head_ops = lm_head_model->get_ordered_ops();
    const auto head_constants = preserve(lm_head_model, true);
    ASSERT_EQ(head_constants.size(), original_constants.size());
    EXPECT_NE(lm_head_model->get_ordered_ops(), head_ops);
    EXPECT_EQ(count_marked_sub128_shifts(lm_head_model), 0u);
    EXPECT_EQ(count_marked_sub128_shifts(model), 2u);
    for (size_t i = 0; i < head_constants.size(); ++i) {
        EXPECT_NE(head_constants[i], original_constants[i]);
        EXPECT_EQ(head_constants[i]->get_element_type(), original_constants[i]->get_element_type());
        EXPECT_EQ(head_constants[i]->get_shape(), original_constants[i]->get_shape());
        EXPECT_EQ(head_constants[i]->get_data_ptr(), original_constants[i]->get_data_ptr());
    }

    ov::TensorVector actual{ov::Tensor(ov::element::f32, ov::Shape{1, kSeq, kHidden}),
                            ov::Tensor(ov::element::f32, ov::Shape{1, kSeq, kHidden})};
    ASSERT_TRUE(model->evaluate(actual, {hidden, ids}));
    ov::TensorVector logits{ov::Tensor(ov::element::f32, ov::Shape{1, kSeq, kVocab})};
    ASSERT_TRUE(lm_head_model->evaluate(logits, {actual[1]}));
    expect_equal_f32_tensors(expected[0], actual[0]);
    expect_equal_f32_tensors(expected[1], logits[0]);

    // In legacy mode the constants are lifted; keep its derived i8 route without an FP16 table unpack.
    for (const auto& constant : preserve(legacy_head)) {
        auto parameter = std::make_shared<ov::op::v0::Parameter>(constant->get_element_type(), constant->get_shape());
        ov::replace_node(constant, parameter);
        legacy_head->add_parameters({parameter});
    }
    Context legacy_ctx(legacy_head, 1);
    ov::pass::GraphRewrite rewrite;
    rewrite.add_matcher<ConvertDQVocab>(std::ref(legacy_ctx));
    rewrite.add_matcher<DQUnpackDictMatMulCWu>(std::ref(legacy_ctx));
    rewrite.run_on_model(legacy_head);
    ASSERT_EQ(legacy_ctx.closures_to_subtract_128.size(), 2u);
    EXPECT_TRUE(legacy_ctx.params_to_unpack.empty());
    for (const auto& [shifted, source] : legacy_ctx.closures_to_subtract_128) {
        EXPECT_EQ(shifted->get_element_type(), ov::element::i8);
        legacy_head->add_parameters({shifted});
        legacy_head->remove_parameter(source);
    }
    EXPECT_NO_THROW(legacy_head->validate_nodes_and_infer_types());
}

INSTANTIATE_TEST_SUITE_P(Precisions,
                         CutLMHeadSharedVocabTest,
                         testing::Values(ov::element::f16, ov::element::f32),
                         [](const testing::TestParamInfo<ov::element::Type>& info) {
                             return info.param.to_string();
                         });

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
