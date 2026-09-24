// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Tests for ov::frontend::gguf::pass::AdaptToGenAI, which rewrites a stateful gguf-IO model
// (inp_tokens/inp_pos/self_kq_mask/token_len_per_seq [+ inp_out_ids, beam_idx]) into genai's
// input_ids/attention_mask/position_ids/beam_idx contract.
//
// These models are built by hand (not via SingleOpBuilder/FrontEnd::convert): AdaptToGenAI needs
// several gguf inputs present together, which no single ggml op translation produces, and the two
// fixes below specifically need the exact node shapes translate_get_rows itself builds.

#include <memory>

#include "common_test_utils/node_builders/constant.hpp"
#include "gtest/gtest.h"
#include "op_test_utils.hpp"
#include "openvino/frontend/gguf/adapt_to_genai.hpp"
#include "openvino/frontend/gguf/make_stateful.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/sink.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/variable.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/sdpa_to_paged_attention.hpp"
#include "utils.hpp"

using namespace ov_gguf_test;
using namespace ov::op;
using ov::frontend::gguf::pass::AdaptToGenAI;

namespace {

// A minimal gguf-IO model: the four Parameters AdaptToGenAI requires, a token-embedding lookup
// built exactly the way translate_get_rows builds it (Squeeze -> Gather -> Unsqueeze, friendly
// name ending "_embd"), and a trivial rank-4 logits Result depending on it. `vocab`/`hidden` size
// the embedding table; `embd` returns the Unsqueeze so tests can inspect/replace it.
//
// If `with_inp_out_ids` is set, also adds an inp_out_ids Parameter and a row-selection subgraph
// on top of embd, built exactly the way translate_get_rows's rank-4/dim1==1 branch builds it
// (Squeeze(embd,[0,1]) -> Gather(., Squeeze(inp_out_ids,[0,1]), axis=0) -> Unsqueeze(axis=0)) --
// the same shape "attn_out_g"/"inpSA_g" have in a real model.
struct MinimalGgufModel {
    std::shared_ptr<ov::Model> model;
    std::shared_ptr<v0::Parameter> inp_tokens;
    std::shared_ptr<ov::Node> embd;  // Unsqueeze(Gather(vocab, Squeeze(inp_tokens)), axis=0)
    std::shared_ptr<v0::Parameter> inp_out_ids;
    std::shared_ptr<ov::Node> row_select;  // Unsqueeze(Gather(Squeeze(embd), Squeeze(inp_out_ids)), axis=0)
    // A second, independent embedding-table lookup keyed on the same inp_tokens indices, mirroring
    // Gemma4's per-layer "pe_tok_flat" (GET_ROWS(per_layer_token_embd.weight, inp_tokens)):
    // Unsqueeze(Gather(pe_table, Squeeze(inp_tokens)), axis=0). Its friendly name deliberately does
    // NOT end in "_embd" -- FixEmbdAxis must find it structurally (rooted at inp_tokens), not by name.
    std::shared_ptr<ov::Node> pe_tok;
};

std::map<std::string, ov::Tensor> make_genai_inputs(size_t length, size_t past = 0) {
    ov::Tensor ids(ov::element::i64, {1, length}), mask(ov::element::i64, {1, past + length});
    ov::Tensor positions(ov::element::i64, {1, length}), beam(ov::element::i32, {1});
    for (size_t i = 0; i < length; ++i) {
        ids.data<int64_t>()[i] = int64_t(i);
        positions.data<int64_t>()[i] = int64_t(past + i);
    }
    std::fill_n(mask.data<int64_t>(), mask.get_size(), 1);
    beam.data<int32_t>()[0] = 0;
    return {{"input_ids", ids}, {"attention_mask", mask}, {"position_ids", positions}, {"beam_idx", beam}};
}

MinimalGgufModel build_minimal_gguf_model(int64_t vocab = 4,
                                          int64_t hidden = 2,
                                          bool with_inp_out_ids = false,
                                          bool with_second_inp_tokens_lookup = false) {
    MinimalGgufModel m;

    m.inp_tokens = ov::test::utils::make_param(ov::element::i32, ov::PartialShape{1, 1, 1, -1}, "inp_tokens");
    auto inp_pos = ov::test::utils::make_param(ov::element::i32, ov::PartialShape{1, 1, 1, -1}, "inp_pos");
    auto self_kq_mask = ov::test::utils::make_param(ov::element::f32, ov::PartialShape{1, 1, -1, -1}, "self_kq_mask");
    auto token_len_per_seq = ov::test::utils::make_param(ov::element::i64, ov::PartialShape{1}, "token_len_per_seq");
    auto beam_idx = ov::test::utils::make_param(ov::element::i32, ov::PartialShape{-1}, "beam_idx");

    std::vector<float> table_values(vocab * hidden);
    for (size_t i = 0; i < table_values.size(); ++i) {
        table_values[i] = static_cast<float>(i);
    }
    auto vocab_table = v0::Constant::create(ov::element::f32, {(size_t)vocab, (size_t)hidden}, table_values);

    auto squeeze_01 = v0::Constant::create(ov::element::i64, {2}, {0, 1});
    auto axis0 = v0::Constant::create(ov::element::i64, {1}, {0});

    // Mirrors translate_get_rows's embedding-table ("else", rank-2 data) branch exactly:
    // Squeeze(indices, [0,1]) -> Gather(table, ., axis=0) -> Unsqueeze(., axis=0).
    auto indices = std::make_shared<v0::Squeeze>(m.inp_tokens, squeeze_01);
    auto gather = std::make_shared<v8::Gather>(vocab_table, indices, axis0);
    m.embd = std::make_shared<v0::Unsqueeze>(gather, axis0);
    m.embd->set_friendly_name("Unsqueeze_test_embd");

    ov::ParameterVector params{m.inp_tokens, inp_pos, self_kq_mask, token_len_per_seq, beam_idx};
    if (with_inp_out_ids) {
        m.inp_out_ids = ov::test::utils::make_param(ov::element::i32, ov::PartialShape{1, 1, 1, -1}, "inp_out_ids");
        params.push_back(m.inp_out_ids);

        // Mirrors translate_get_rows's rank-4/dim1==1 branch ("attn_out_g"/"inpSA_g" in a real
        // model): Squeeze(data,[0,1]) -> Gather(., Squeeze(indices,[0,1]), axis=0) -> Unsqueeze(0).
        auto data_squeeze = std::make_shared<v0::Squeeze>(m.embd, squeeze_01);
        auto row_indices = std::make_shared<v0::Squeeze>(m.inp_out_ids, squeeze_01);
        auto row_gather = std::make_shared<v8::Gather>(data_squeeze, row_indices, axis0);
        m.row_select = std::make_shared<v0::Unsqueeze>(row_gather, axis0);
        m.row_select->set_friendly_name("Unsqueeze_test_row_select");
    }

    ov::ResultVector results;
    if (with_second_inp_tokens_lookup) {
        std::vector<float> pe_table_values(vocab * hidden);
        for (size_t i = 0; i < pe_table_values.size(); ++i) {
            pe_table_values[i] = static_cast<float>(1000 + i);
        }
        auto pe_table = v0::Constant::create(ov::element::f32, {(size_t)vocab, (size_t)hidden}, pe_table_values);
        auto pe_indices = std::make_shared<v0::Squeeze>(m.inp_tokens, squeeze_01);
        auto pe_gather = std::make_shared<v8::Gather>(pe_table, pe_indices, axis0);
        m.pe_tok = std::make_shared<v0::Unsqueeze>(pe_gather, axis0);
        m.pe_tok->set_friendly_name("Unsqueeze_test_pe_tok_flat");
        // Reachable from a Result: MatcherPass only visits nodes reachable from the model's
        // results/sinks, so an orphan branch would silently never be matched.
        results.push_back(std::make_shared<v0::Result>(m.pe_tok));
    }

    // A trivial rank-4 "logits" output so AdaptToGenAI's final logits reshape has something to
    // work on: reduce the hidden axis so the shape stays predictable regardless of vocab/hidden.
    auto logits_src = with_inp_out_ids ? m.row_select : m.embd;
    auto reduce_axis_3 = v0::Constant::create(ov::element::i64, {1}, {3});
    auto logits = std::make_shared<v1::ReduceSum>(logits_src, reduce_axis_3, true);
    results.insert(results.begin(), std::make_shared<v0::Result>(logits));

    m.model = std::make_shared<ov::Model>(results, params);
    return m;
}

}  // namespace

// AdaptToGenAI rewrites the gguf-style IO into genai's input_ids/attention_mask/position_ids
// contract, and beam_idx passes through unchanged (genai sets it directly).
TEST(GGUFAdaptToGenAI, RewritesIOContract) {
    auto m = build_minimal_gguf_model();

    ASSERT_TRUE(AdaptToGenAI().run_on_model(m.model));

    auto input_ids = find_parameter(m.model, "input_ids");
    auto attention_mask = find_parameter(m.model, "attention_mask");
    auto position_ids = find_parameter(m.model, "position_ids");
    auto beam_idx = find_parameter(m.model, "beam_idx");
    ASSERT_NE(input_ids, nullptr);
    ASSERT_NE(attention_mask, nullptr);
    ASSERT_NE(position_ids, nullptr);
    ASSERT_NE(beam_idx, nullptr);
    EXPECT_EQ(input_ids->get_element_type(), ov::element::i64);
    EXPECT_EQ(input_ids->get_partial_shape(), ov::PartialShape({-1, -1}));

    // logits reshaped to rank 3 [batch, seq, vocab].
    ASSERT_EQ(m.model->get_results().size(), 1);
    EXPECT_EQ(m.model->get_results()[0]->get_output_partial_shape(0).rank().get_length(), 3);
}

TEST(GGUFAdaptToGenAI, AcceptsPrunedTokenCountInput) {
    auto m = build_minimal_gguf_model();
    auto count = find_parameter(m.model, "token_len_per_seq");
    ASSERT_NE(count, nullptr);
    ASSERT_TRUE(count->output(0).get_target_inputs().empty());
    m.model->remove_parameter(count);

    ASSERT_TRUE(AdaptToGenAI().run_on_model(m.model));
    EXPECT_NE(find_parameter(m.model, "input_ids"), nullptr);
    EXPECT_NE(find_parameter(m.model, "attention_mask"), nullptr);
    EXPECT_NE(find_parameter(m.model, "position_ids"), nullptr);
    EXPECT_EQ(m.model->get_results().size(), 1);
    EXPECT_EQ(m.model->output().get_partial_shape().rank().get_length(), 3);
    EXPECT_FALSE(AdaptToGenAI().run_on_model(m.model));
}

// Without the required gguf inputs present, the pass is a no-op (e.g. a model already adapted, or
// not a gguf-IO model at all).
TEST(GGUFAdaptToGenAI, NoOpWithoutGgufInputs) {
    auto input_ids = std::make_shared<v0::Parameter>(ov::element::i64, ov::PartialShape{-1, -1});
    auto result = std::make_shared<v0::Result>(input_ids);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input_ids});

    EXPECT_FALSE(AdaptToGenAI().run_on_model(model));
}

TEST(GGUFAdaptToGenAI, EmbeddingModeExtractsLookupAndAcceptsInjectedValues) {
    auto m = build_minimal_gguf_model();
    m.embd->get_rt_info()["gguf.token_embedding"] = true;
    // Keep an unreachable consumer alive across adaptation. Rewiring inp_tokens also
    // updates this node, but it must not keep input_ids in the decoder's input contract.
    auto detached = std::make_shared<v0::Convert>(m.inp_tokens, ov::element::i64);
    AdaptToGenAI pass(AdaptToGenAI::InputMode::EMBEDS_TO_LOGITS);
    ASSERT_TRUE(pass.run_on_model(m.model));
    ASSERT_NE(pass.get_embedding_model(), nullptr);
    EXPECT_EQ(find_parameter(m.model, "input_ids"), nullptr);
    ASSERT_NE(find_parameter(m.model, "inputs_embeds"), nullptr);
    EXPECT_NE(detached->input_value(0).get_node(), m.inp_tokens.get());
    for (size_t length : {1, 3}) {
        ov::Tensor ids(ov::element::i64, {1, length});
        for (size_t i = 0; i < length; ++i)
            ids.data<int64_t>()[i] = int64_t(i);
        ov::TensorVector lookup{ov::Tensor(ov::element::f32, {1, length, 2})};
        ASSERT_TRUE(pass.get_embedding_model()->evaluate(lookup, {ids}));
        for (size_t i = 0; i < 2 * length; ++i)
            EXPECT_EQ(lookup[0].data<float>()[i], float(i));
        ov::TensorVector inputs;
        for (const auto& p : m.model->get_parameters()) {
            if (p->get_friendly_name() == "inputs_embeds") {
                inputs.push_back(lookup[0]);
            } else {
                const auto shape = p->get_friendly_name() == "beam_idx" ? ov::Shape{1} : ov::Shape{1, length};
                inputs.emplace_back(p->get_element_type(), shape);
                std::memset(inputs.back().data(), 0, inputs.back().get_byte_size());
            }
        }
        ov::TensorVector outputs{ov::Tensor(ov::element::f32, {1, length, 1})};
        ASSERT_TRUE(m.model->evaluate(outputs, inputs));
        ASSERT_EQ(outputs[0].get_shape(), (ov::Shape{1, length, 1}));
        for (size_t i = 0; i < length; ++i)
            EXPECT_EQ(outputs[0].data<float>()[i], float(4 * i + 1));
    }
    ov::Tensor batch_ids(ov::element::i64, {2, 1});
    batch_ids.data<int64_t>()[0] = 1;
    batch_ids.data<int64_t>()[1] = 3;
    ov::TensorVector batch_embeddings{ov::Tensor(ov::element::f32, {2, 1, 2})};
    ASSERT_TRUE(pass.get_embedding_model()->evaluate(batch_embeddings, {batch_ids}));
    EXPECT_EQ(batch_embeddings[0].get_shape(), (ov::Shape{2, 1, 2}));
    EXPECT_EQ(batch_embeddings[0].data<float>()[0], 2.f);
    EXPECT_EQ(batch_embeddings[0].data<float>()[2], 6.f);
    EXPECT_FALSE(pass.run_on_model(m.model));
}

TEST(GGUFAdaptToGenAI, EmbeddingModeMovesPerLayerTokenLookupToEmbeddingModel) {
    auto m = build_minimal_gguf_model(4, 2, false, true);
    m.embd->get_rt_info()["gguf.token_embedding"] = true;
    m.pe_tok->get_rt_info()["gguf.per_layer_token_embedding"] = int64_t{2};
    AdaptToGenAI pass(AdaptToGenAI::InputMode::EMBEDS_TO_LOGITS);
    ASSERT_TRUE(pass.run_on_model(m.model));
    EXPECT_EQ(find_parameter(m.model, "input_ids"), nullptr);
    EXPECT_NE(find_parameter(m.model, "inputs_embeds"), nullptr);
    EXPECT_EQ(m.model->input("per_layer_inputs").get_partial_shape(), (ov::PartialShape{-1, -1, 2, 1}));
    const auto& lookup = pass.get_embedding_model();
    EXPECT_EQ(lookup->get_parameters().size(), 1);
    ASSERT_EQ(lookup->outputs().size(), 2);
    EXPECT_EQ(lookup->output(0).get_any_name(), "inputs_embeds");
    EXPECT_EQ(lookup->output(1).get_any_name(), "per_layer_inputs");
    EXPECT_EQ(lookup->output(1).get_partial_shape(), (ov::PartialShape{-1, -1, 2, 1}));
}

// An untagged token lookup cannot be moved, so the language model keeps input_ids for it.
TEST(GGUFAdaptToGenAI, EmbeddingModePreservesUntaggedAuxiliaryTokenLookup) {
    auto m = build_minimal_gguf_model(4, 2, false, true);
    m.embd->get_rt_info()["gguf.token_embedding"] = true;
    AdaptToGenAI pass(AdaptToGenAI::InputMode::EMBEDS_TO_LOGITS);
    ASSERT_TRUE(pass.run_on_model(m.model));
    EXPECT_NE(find_parameter(m.model, "input_ids"), nullptr);
    EXPECT_NE(find_parameter(m.model, "inputs_embeds"), nullptr);
    EXPECT_EQ(pass.get_embedding_model()->get_parameters().size(), 1);
}

TEST(GGUFAdaptToGenAI, EmbeddingModeRetainsScalingOnce) {
    auto m = build_minimal_gguf_model();
    m.embd->get_rt_info()["gguf.token_embedding"] = true;
    auto reduction = m.model->get_results().front()->input_value(0).get_node_shared_ptr();
    reduction->input(0).replace_source_output(
        std::make_shared<v1::Multiply>(m.embd, v0::Constant::create(ov::element::f32, {}, {7.f})));
    auto reference = m.model->clone();
    AdaptToGenAI().run_on_model(reference);
    AdaptToGenAI pass(AdaptToGenAI::InputMode::EMBEDS_TO_LOGITS);
    pass.run_on_model(m.model);
    auto inputs = make_genai_inputs(3);
    auto expected = run_on_cpu(reference, inputs);
    auto lookup = run_on_cpu(pass.get_embedding_model(), {{"input_ids", inputs.at("input_ids")}});
    inputs.erase("input_ids");
    inputs.emplace("inputs_embeds", lookup);
    auto actual = run_on_cpu(m.model, inputs);
    ASSERT_EQ(actual.get_shape(), expected.get_shape());
    for (size_t i = 0; i < actual.get_size(); ++i)
        EXPECT_FLOAT_EQ(actual.data<float>()[i], expected.data<float>()[i]);
}

TEST(GGUFAdaptToGenAI, BatchedMaskKeepsSequencesSeparateAndExcludesPadding) {
    auto m = build_minimal_gguf_model();
    auto mask = find_parameter(m.model, "self_kq_mask");
    m.model->add_results({std::make_shared<v0::Result>(mask)});
    ASSERT_TRUE(AdaptToGenAI().run_on_model(m.model));
    ov::Core core;
    auto request = core.compile_model(m.model, "CPU").create_infer_request();
    // The second sequence has two left-padding columns and one current token.
    for (size_t past : {0, 2}) {
        ov::Tensor ids(ov::element::i64, {2, 3}), positions(ov::element::i64, {2, 3});
        ov::Tensor attention(ov::element::i64, {2, past + 3}), beams(ov::element::i32, {2});
        std::fill_n(ids.data<int64_t>(), ids.get_size(), 1);
        std::fill_n(attention.data<int64_t>(), attention.get_size(), 1);
        attention.data<int64_t>()[past + 3] = attention.data<int64_t>()[past + 4] = 0;
        for (size_t i = 0; i < 3; ++i) {
            positions.data<int64_t>()[i] = past + i;
            positions.data<int64_t>()[3 + i] = std::max<int64_t>(0, int64_t(past + i) - 2);
        }
        beams.data<int32_t>()[0] = 0;
        beams.data<int32_t>()[1] = 1;
        request.set_tensor("input_ids", ids);
        request.set_tensor("attention_mask", attention);
        request.set_tensor("position_ids", positions);
        request.set_tensor("beam_idx", beams);
        request.infer();
        const auto actual = request.get_output_tensor(0);
        ASSERT_EQ(actual.get_shape(), (ov::Shape{2, 1, 3, past + 3}));
        for (size_t b = 0; b < 2; ++b) {
            for (size_t q = 0; q < 3; ++q) {
                for (size_t k = 0; k < past + 3; ++k) {
                    const bool allowed = k <= past + q && (b == 0 || k >= 2);
                    const float value = actual.data<const float>()[(b * 3 + q) * (past + 3) + k];
                    // Padded query rows are ignored by generation.
                    if (b == 1 && past + q < 2)
                        continue;
                    if (allowed)
                        EXPECT_EQ(value, 0.f);
                    else
                        EXPECT_LT(value, -1e4f);
                }
            }
        }
    }
}

class GGUFAdaptToGenAIImageMask : public testing::TestWithParam<std::string> {};

TEST_P(GGUFAdaptToGenAIImageMask, RespectsLayerTypeImageGroupsAndCachedPrefix) {
    auto m = build_minimal_gguf_model();
    m.embd->get_rt_info()["gguf.token_embedding"] = true;
    m.model->get_rt_info()["gguf_architecture"] = GetParam();
    m.model->get_rt_info()[ov::frontend::gguf::pass::gguf_swa_window_key()] = int64_t{3};
    auto mask = find_parameter(m.model, "self_kq_mask");
    auto swa_mask = ov::test::utils::make_param(ov::element::f32,
                                               ov::PartialShape{1, 1, -1, -1},
                                               "self_kq_mask_swa");
    m.model->add_parameters({swa_mask});
    m.model->add_results({std::make_shared<v0::Result>(mask), std::make_shared<v0::Result>(swa_mask)});
    AdaptToGenAI(AdaptToGenAI::InputMode::EMBEDS_TO_LOGITS).run_on_model(m.model);
    // Adaptation replaces the original first result with logits; the retained mask is now first.
    ov::Core core;
    auto request = core.compile_model(m.model, "CPU").create_infer_request();
    const std::vector<int64_t> types{0, 1, 1, 0, 1, 1, 0};
    for (size_t past : {0, 3}) {
        auto inputs = make_genai_inputs(types.size(), past);
        inputs.erase("input_ids");
        inputs.emplace("inputs_embeds", ov::Tensor(ov::element::f32, {1, types.size(), 2}));
        ov::Tensor token_types(ov::element::i64, {1, types.size()});
        std::copy(types.begin(), types.end(), token_types.data<int64_t>());
        inputs.emplace("token_type_ids", token_types);
        for (auto& entry : inputs)
            request.set_tensor(entry.first, entry.second);
        request.infer();
        for (size_t layer = 0; layer < 2; ++layer) {
            auto actual = request.get_output_tensor(layer);
            ASSERT_EQ(actual.get_shape(), (ov::Shape{1, 1, types.size(), past + types.size()}));
            const bool bidirectional = layer == 1 || GetParam() == "gemma3";
            for (size_t q = 0; q < types.size(); ++q) {
                for (size_t k = 0; k < past + types.size(); ++k) {
                    const bool same_image = k >= past && types[q] == 1 && types[k - past] == 1 &&
                                            ((q <= 2 && k - past <= 2) || (q >= 4 && k - past >= 4));
                    const bool allowed = (k <= past + q || (bidirectional && same_image)) &&
                                         (layer == 0 || k + 3 > past + q);
                    const float value = actual.data<float>()[q * (past + types.size()) + k];
                    if (allowed)
                        EXPECT_EQ(value, 0.f) << "layer=" << layer << " past=" << past << " q=" << q << " k=" << k;
                    else
                        EXPECT_LT(value, -1e4f) << "layer=" << layer << " past=" << past << " q=" << q << " k=" << k;
                }
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(Gemma,
                         GGUFAdaptToGenAIImageMask,
                         testing::Values(std::string("gemma3"), std::string("gemma4")),
                         [](const testing::TestParamInfo<std::string>& info) { return info.param; });

TEST(GGUFAdaptToGenAI, EmbeddingModeMapsGenAIMultimodalPositionsToGGML) {
    auto m = build_minimal_gguf_model();
    m.embd->get_rt_info()["gguf.token_embedding"] = true;
    m.model->get_rt_info()[ov::frontend::gguf::pass::gguf_imrope_key()] = true;
    auto positions = find_parameter(m.model, "inp_pos");
    m.model->add_results({std::make_shared<v0::Result>(positions)});
    AdaptToGenAI(AdaptToGenAI::InputMode::EMBEDS_TO_LOGITS).run_on_model(m.model);
    auto inputs = make_genai_inputs(2, 7);
    inputs.erase("input_ids");
    inputs.emplace("inputs_embeds", make_f32_tensor({1, 2, 2}, {1, 2, 3, 4}));
    ov::Tensor coordinates(ov::element::i64, {4, 1, 2});
    const std::vector<int64_t> data{3, 7, 11, 2, 5, 13, 17, 19};
    std::copy(data.begin(), data.end(), coordinates.data<int64_t>());
    inputs["position_ids"] = coordinates;
    auto actual = run_on_cpu(m.model, inputs);
    ASSERT_EQ(actual.get_shape(), (ov::Shape{1, 1, 1, 8}));
    for (size_t i = 0; i < data.size(); ++i)
        EXPECT_EQ(actual.data<int32_t>()[i], data[(i + 2) % data.size()]);
}

// translate_get_rows's embedding lookup restores ggml's rank-4 form with Unsqueeze(axis=0),
// pinning the leading axis to a literal 1. AdaptToGenAI moves it to axis 1 so it instead mirrors
// input_ids' own (backend-dependent) leading dim -- see FixEmbdAxis in adapt_to_genai.cpp.
TEST(GGUFAdaptToGenAI, FixesEmbdAxisStructurally) {
    auto m = build_minimal_gguf_model();
    auto embd_gather_input = m.embd->input_value(0);  // survives the Unsqueeze replacement

    ASSERT_TRUE(AdaptToGenAI().run_on_model(m.model));

    // The original Unsqueeze(axis=0) node was replaced; find its successor by name (the
    // replacement copies it) and confirm the new axis is 1, fed by the same Gather.
    std::shared_ptr<ov::Node> fixed_embd;
    for (const auto& op : m.model->get_ordered_ops()) {
        if (op->get_friendly_name() == "Unsqueeze_test_embd") {
            fixed_embd = op;
            break;
        }
    }
    ASSERT_NE(fixed_embd, nullptr);
    ASSERT_EQ(fixed_embd->get_type_name(), std::string("Unsqueeze"));
    EXPECT_EQ(fixed_embd->input_value(0), embd_gather_input);  // still the same Gather, unchanged

    auto axis_const = ov::as_type_ptr<v0::Constant>(fixed_embd->input_value(1).get_node_shared_ptr());
    ASSERT_NE(axis_const, nullptr);
    EXPECT_EQ(axis_const->cast_vector<int64_t>(), std::vector<int64_t>({1}));
}

// Regression test for the BLOCKER FixEmbdAxis's name-only matcher used to miss: Gemma4 has a
// SECOND embedding-table lookup rooted at inp_tokens (GET_ROWS(per_layer_token_embd.weight,
// inp_tokens) -> "pe_tok_flat"), alongside the main "embd" lookup (GET_ROWS(token_embd.weight,
// inp_tokens)). Both must move their leading axis from 0 to 1 identically; otherwise the two
// disagree on which axis holds the leading placeholder, and their eventual sum (per_layer_embd =
// pe_proj + pe_tok, or ffn_inp = attn_out + embd) broadcasts to a spurious [tokens, tokens, ...]
// shape under PagedAttention instead of [tokens, ...]. FixEmbdAxis must therefore match every
// embedding-table GET_ROWS structurally rooted at inp_tokens, not just the one whose friendly name
// happens to end in "_embd".
TEST(GGUFAdaptToGenAI, FixesEmbdAxisStructurallyForSecondaryInpTokensLookup) {
    auto m = build_minimal_gguf_model(/*vocab=*/4,
                                      /*hidden=*/2,
                                      /*with_inp_out_ids=*/false,
                                      /*with_second_inp_tokens_lookup=*/true);
    ASSERT_NE(m.pe_tok, nullptr);
    auto pe_tok_gather_input = m.pe_tok->input_value(0);  // survives the Unsqueeze replacement

    ASSERT_TRUE(AdaptToGenAI().run_on_model(m.model));

    std::shared_ptr<ov::Node> fixed_pe_tok;
    for (const auto& op : m.model->get_ordered_ops()) {
        if (op->get_friendly_name() == "Unsqueeze_test_pe_tok_flat") {
            fixed_pe_tok = op;
            break;
        }
    }
    ASSERT_NE(fixed_pe_tok, nullptr);
    ASSERT_EQ(fixed_pe_tok->get_type_name(), std::string("Unsqueeze"));
    EXPECT_EQ(fixed_pe_tok->input_value(0), pe_tok_gather_input);  // still the same Gather, unchanged

    auto axis_const = ov::as_type_ptr<v0::Constant>(fixed_pe_tok->input_value(1).get_node_shared_ptr());
    ASSERT_NE(axis_const, nullptr);
    // Same axis (1) as embd's own fix, above -- the two branches must agree.
    EXPECT_EQ(axis_const->cast_vector<int64_t>(), std::vector<int64_t>({1}));
}

// Functional regression test for the bug FixEmbdAxis fixes: under SDPAToPagedAttention,
// input_ids' own leading dim flips from batch (pre-PA, [1, tokens]) to tokens (post-PA rewrite,
// [tokens, 1]). embd's leading axis must flip the same way -- 1 pre-PA, tokens post-PA -- instead
// of staying pinned at 1 regardless, which used to silently fold every extra token into embd's
// trailing (feature) dimension.
TEST(GGUFAdaptToGenAI, EmbdLeadingAxisSelfCorrectsUnderBothLayouts) {
    const int64_t hidden = 3;
    auto m = build_minimal_gguf_model(/*vocab=*/8, hidden);
    // Add the probe Result *before* running the pass: FixEmbdAxis replaces embd's Unsqueeze, and
    // Output::replace() only rewires consumers that already exist at that point.
    auto embd_result = std::make_shared<v0::Result>(m.embd);
    m.model->add_results({embd_result});
    ASSERT_TRUE(AdaptToGenAI().run_on_model(m.model));
    m.model->validate_nodes_and_infer_types();

    ov::Core core;
    auto compiled = core.compile_model(m.model, "CPU");
    auto req = compiled.create_infer_request();

    const std::vector<int64_t> tokens{2, 5, 7};
    const size_t T = tokens.size();

    // Pre-PA layout: input_ids [1, tokens] -- axis 1 lands right where the old, unfixed
    // Unsqueeze(axis=0) already put it (indices' own leading axis is already 1 here), so this
    // layout's shape is numerically unchanged from before the fix.
    {
        ov::Tensor t(ov::element::i64, {1, T});
        std::copy(tokens.begin(), tokens.end(), t.data<int64_t>());
        req.set_tensor("input_ids", t);
        req.infer();
        auto embd_out = req.get_tensor(embd_result->output(0));
        EXPECT_EQ(embd_out.get_shape(), ov::Shape({1, 1, T, (size_t)hidden}));
    }

    // Post-PA layout: SDPAToPagedAttention rewrites input_ids to rank-1 [tokens] and splices an
    // Unsqueeze(-1) in front of consumers, so the body sees [tokens, 1]. Feed that shape directly
    // (equivalent to what the rewritten graph would compute) without needing to run the actual
    // SDPAToPagedAttention pass.
    {
        ov::Tensor t(ov::element::i64, {T, 1});
        std::copy(tokens.begin(), tokens.end(), t.data<int64_t>());
        req.set_tensor("input_ids", t);
        req.infer();
        auto embd_out = req.get_tensor(embd_result->output(0));
        // Self-correcting: the leading axis is now T (the real token count), not 1.
        EXPECT_EQ(embd_out.get_shape(), ov::Shape({T, 1, 1, (size_t)hidden}));

        // And the values themselves are unaffected by the reshape -- still exactly the embedding
        // table's rows for these tokens, just addressable at a different leading axis.
        const float* data = embd_out.data<float>();
        for (size_t i = 0; i < T; ++i) {
            for (int64_t h = 0; h < hidden; ++h) {
                EXPECT_FLOAT_EQ(data[i * hidden + h], static_cast<float>(tokens[i] * hidden + h));
            }
        }
    }
}

// translate_get_rows's inp_out_ids row-selection (attn_out_g/inpSA_g/...) uses a batch_dims=1
// Gather keyed on the residual stream's leading axis staying literal 1. FixInpOutIdsRowSelect
// replaces it with a flatten-based selection that works regardless of which axis holds the real
// token count -- see FixInpOutIdsRowSelect in adapt_to_genai.cpp.
TEST(GGUFAdaptToGenAI, FixesInpOutIdsRowSelectStructurally) {
    auto m = build_minimal_gguf_model(/*vocab=*/4, /*hidden=*/2, /*with_inp_out_ids=*/true);

    ASSERT_TRUE(AdaptToGenAI().run_on_model(m.model));

    std::shared_ptr<ov::Node> fixed_row_select;
    std::shared_ptr<ov::Node> fixed_embd;
    for (const auto& op : m.model->get_ordered_ops()) {
        if (op->get_friendly_name() == "Unsqueeze_test_row_select") {
            fixed_row_select = op;
        } else if (op->get_friendly_name() == "Unsqueeze_test_embd") {
            fixed_embd = op;  // the *replaced* embd node -- m.embd itself is now a stale, orphaned copy
        }
    }
    ASSERT_NE(fixed_row_select, nullptr);
    ASSERT_NE(fixed_embd, nullptr);
    ASSERT_EQ(fixed_row_select->get_type_name(), std::string("Unsqueeze"));

    auto gather = ov::as_type_ptr<v8::Gather>(fixed_row_select->input_value(0).get_node_shared_ptr());
    ASSERT_NE(gather, nullptr);

    // Both the activation and the indices are now fed through a Reshape (flatten), not a Squeeze.
    auto data_flat = ov::as_type_ptr<v1::Reshape>(gather->input_value(0).get_node_shared_ptr());
    auto indices_flat = ov::as_type_ptr<v1::Reshape>(gather->input_value(1).get_node_shared_ptr());
    ASSERT_NE(data_flat, nullptr);
    ASSERT_NE(indices_flat, nullptr);
    EXPECT_EQ(data_flat->input_value(0), fixed_embd->output(0));  // still fed by the (now-fixed) embd
}

// Functional regression test: under the pre-PA layout ([1, tokens]), FixInpOutIdsRowSelect must
// still select just the last token's row (the row-selection's whole point -- skip projecting
// every other prompt position to vocab). Under the post-PA layout ([tokens, 1]) each row already
// holds exactly one token, so "last row within a row" is the identity: all tokens pass through
// unchanged, and genai's own PagedAttention machinery -- not this graph -- picks which ones matter.
TEST(GGUFAdaptToGenAI, InpOutIdsRowSelectionCorrectUnderBothLayouts) {
    const int64_t hidden = 3;
    auto m = build_minimal_gguf_model(/*vocab=*/8, hidden, /*with_inp_out_ids=*/true);
    // Add the probe Result before running the pass, same reasoning as the embd functional test.
    auto row_result = std::make_shared<v0::Result>(m.row_select);
    m.model->add_results({row_result});
    ASSERT_TRUE(AdaptToGenAI().run_on_model(m.model));
    m.model->validate_nodes_and_infer_types();

    ov::Core core;
    auto compiled = core.compile_model(m.model, "CPU");
    auto req = compiled.create_infer_request();

    const std::vector<int64_t> tokens{2, 5, 7};
    const size_t T = tokens.size();

    // Pre-PA layout: input_ids [1, tokens] -- only the last token's row survives.
    {
        ov::Tensor t(ov::element::i64, {1, T});
        std::copy(tokens.begin(), tokens.end(), t.data<int64_t>());
        req.set_tensor("input_ids", t);
        req.infer();
        auto row_out = req.get_tensor(row_result->output(0));

        ASSERT_EQ(row_out.get_size(), (size_t)hidden);
        const float* data = row_out.data<float>();
        for (int64_t h = 0; h < hidden; ++h) {
            EXPECT_FLOAT_EQ(data[h], static_cast<float>(tokens.back() * hidden + h));
        }
    }

    // Post-PA layout: input_ids [tokens, 1] -- every row already holds one token, so all T rows
    // pass through unchanged, in order.
    {
        ov::Tensor t(ov::element::i64, {T, 1});
        std::copy(tokens.begin(), tokens.end(), t.data<int64_t>());
        req.set_tensor("input_ids", t);
        req.infer();
        auto row_out = req.get_tensor(row_result->output(0));

        ASSERT_EQ(row_out.get_size(), T * (size_t)hidden);
        const float* data = row_out.data<float>();
        for (size_t i = 0; i < T; ++i) {
            for (int64_t h = 0; h < hidden; ++h) {
                EXPECT_FLOAT_EQ(data[i * hidden + h], static_cast<float>(tokens[i] * hidden + h));
            }
        }
    }
}

// Functional/structural regression test for the whole bug class FixEmbdAxis and
// FixInpOutIdsRowSelect fix: a genai-adapted model whose residual stream (embd) feeds a REAL
// attention block (Q/K/V projections, a stateful KV cache, and an actual
// v13::ScaledDotProductAttention) must survive ov::pass::SDPAToPagedAttention -- the exact
// transformation that rewrites input_ids to rank-1 [tokens] and is what actually exposed this bug
// (see the class comment on FixEmbdAxis in adapt_to_genai.cpp). Unlike the tests above, which probe
// embd/row-selection in isolation and hand-feed a [tokens, 1] tensor to simulate the post-PA
// layout, this test lets SDPAToPagedAttention itself perform that rewrite and drives Q all the way
// through a real PagedAttentionExtension node.
//
// Heads=1 and head_size=hidden keep the head-merge trivial. Q/K project through an all-zero
// weight matrix (so every attention score is 0, i.e. uniform weights over the causal window) and V
// projects through the identity matrix (so V's rows are exactly embd's own rows); with a full
// causal mask this makes the attention output for token i exactly mean(embedding_row(token_0..i))
// -- directly exercising whether embd's leading axis threaded correctly into Q/K/V/PagedAttention
// per token, without needing real RoPE weights or a llama.cpp oracle to compute an expected value.
namespace {

std::shared_ptr<ov::Model> build_attention_gguf_model(int64_t vocab, int64_t hidden, bool auxiliary_tokens = false) {
    auto inp_tokens = ov::test::utils::make_param(ov::element::i32, ov::PartialShape{1, 1, 1, -1}, "inp_tokens");
    auto inp_pos = ov::test::utils::make_param(ov::element::i32, ov::PartialShape{1, 1, 1, -1}, "inp_pos");
    auto self_kq_mask = ov::test::utils::make_param(ov::element::f32, ov::PartialShape{1, 1, -1, -1}, "self_kq_mask");
    auto token_len_per_seq = ov::test::utils::make_param(ov::element::i64, ov::PartialShape{1}, "token_len_per_seq");
    auto beam_idx = ov::test::utils::make_param(ov::element::i32, ov::PartialShape{-1}, "beam_idx");

    // Token embedding table: row i is filled with value i (so the expected running-mean output
    // is trivial to compute from the token ids alone).
    std::vector<float> table_values(vocab * hidden);
    for (int64_t v = 0; v < vocab; ++v) {
        for (int64_t h = 0; h < hidden; ++h) {
            table_values[v * hidden + h] = static_cast<float>(v);
        }
    }
    auto vocab_table = v0::Constant::create(ov::element::f32, {(size_t)vocab, (size_t)hidden}, table_values);

    auto squeeze_01 = v0::Constant::create(ov::element::i64, {2}, {0, 1});
    auto axis0 = v0::Constant::create(ov::element::i64, {1}, {0});

    // Mirrors translate_get_rows's embedding-table branch exactly, same as build_minimal_gguf_model.
    auto indices = std::make_shared<v0::Squeeze>(inp_tokens, squeeze_01);
    auto gather = std::make_shared<v8::Gather>(vocab_table, indices, axis0);
    auto embd = std::make_shared<v0::Unsqueeze>(gather, axis0);
    embd->set_friendly_name("embd");
    embd->get_rt_info()["gguf.token_embedding"] = true;

    // Flatten to [1, tokens, hidden] regardless of which axis (0 pre-fix, 1 post-fix) carries the
    // real token count -- Reshape never reorders memory, so this is correct either way (same trick
    // FixInpOutIdsRowSelect itself uses).
    auto embd_3d_shape = v0::Constant::create(ov::element::i64, {3}, std::vector<int64_t>{1, -1, hidden});
    ov::Output<ov::Node> combined = embd;
    if (auxiliary_tokens) {
        auto auxiliary = std::make_shared<v8::Gather>(vocab_table, indices, axis0);
        auto lifted = std::make_shared<v0::Unsqueeze>(auxiliary, axis0);
        lifted->set_friendly_name("per_layer_tokens");
        // Two layers of hidden / 2 values each, as Gemma4's per-layer token embedding.
        lifted->get_rt_info()["gguf.per_layer_token_embedding"] = int64_t{2};
        combined = std::make_shared<v1::Add>(embd, lifted);
    }
    auto embd_3d = std::make_shared<v1::Reshape>(combined, embd_3d_shape, false);

    std::vector<float> zero_w(hidden * hidden, 0.0f);
    auto w_zero = v0::Constant::create(ov::element::f32, {(size_t)hidden, (size_t)hidden}, zero_w);
    std::vector<float> identity_w(hidden * hidden, 0.0f);
    for (int64_t i = 0; i < hidden; ++i) {
        identity_w[i * hidden + i] = 1.0f;
    }
    auto w_identity = v0::Constant::create(ov::element::f32, {(size_t)hidden, (size_t)hidden}, identity_w);

    // Single-head Q/K/V projection: MatMul -> reshape to [1, tokens, 1, head_size] -> transpose to
    // [1, 1, tokens, head_size].
    auto transpose_0213 = v0::Constant::create(ov::element::i64, {4}, {0, 2, 1, 3});
    auto make_projection = [&](const std::shared_ptr<v0::Constant>& weight) {
        auto matmul = std::make_shared<v0::MatMul>(embd_3d, weight, false, true);
        auto heads_shape = v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{0, 0, 1, hidden});
        auto heads = std::make_shared<v1::Reshape>(matmul, heads_shape, true);
        return std::make_shared<v1::Transpose>(heads, transpose_0213);
    };
    auto q = make_projection(w_zero);
    auto k = make_projection(w_zero);
    auto v = make_projection(w_identity);

    // Stateful KV cache: ReadValue -> Gather(beam_idx) -> Concat(-2, cur) -> Assign, exactly what
    // ov::frontend::gguf::pass::MakeStateful emits for a real model.
    auto make_kv_cache = [&](const ov::Output<ov::Node>& cur, const std::string& var_id) {
        auto var = std::make_shared<util::Variable>(
            util::VariableInfo{ov::PartialShape{-1, 1, -1, hidden}, ov::element::f32, var_id});
        auto init_shape = v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{1, 1, 0, hidden});
        auto init = std::make_shared<v3::Broadcast>(v0::Constant::create(ov::element::f32, {}, {0.0f}), init_shape);
        auto read = std::make_shared<v6::ReadValue>(init, var);
        auto past = std::make_shared<v8::Gather>(read, beam_idx, axis0, 0);
        auto concat = std::make_shared<v0::Concat>(ov::OutputVector{past, cur}, -2);
        auto assign = std::make_shared<v6::Assign>(concat, var);
        return std::make_pair(ov::Output<ov::Node>(concat), std::static_pointer_cast<Sink>(assign));
    };
    auto [k_concat, k_assign] = make_kv_cache(k, "attn_k_cache.0");
    auto [v_concat, v_assign] = make_kv_cache(v, "attn_v_cache.0");

    auto scale = v0::Constant::create(ov::element::f32, {}, {1.0f});
    auto sdpa = std::make_shared<v13::ScaledDotProductAttention>(q,
                                                                 k_concat,
                                                                 v_concat,
                                                                 self_kq_mask,
                                                                 scale,
                                                                 /*causal=*/false);

    // Merge heads back and flatten to [1, tokens, hidden] -- the "logits"-shaped output
    // AdaptToGenAI's final reshape expects.
    auto merged = std::make_shared<v1::Transpose>(sdpa, transpose_0213);
    auto merged_flat_shape = v0::Constant::create(ov::element::i64, {3}, {0, 0, -1});
    auto merged_flat = std::make_shared<v1::Reshape>(merged, merged_flat_shape, true);
    auto result = std::make_shared<v0::Result>(merged_flat);

    return std::make_shared<ov::Model>(
        ov::ResultVector{result},
        ov::SinkVector{k_assign, v_assign},
        ov::ParameterVector{inp_tokens, inp_pos, self_kq_mask, token_len_per_seq, beam_idx});
}

}  // namespace

TEST(GGUFAdaptToGenAI, SurvivesSDPAToPagedAttentionWithRealAttentionBlock) {
    const int64_t vocab = 8;
    const int64_t hidden = 4;
    auto model = build_attention_gguf_model(vocab, hidden);

    ASSERT_TRUE(AdaptToGenAI().run_on_model(model));
    ASSERT_NO_THROW(model->validate_nodes_and_infer_types());

    // Run the real transformation this bug was found under -- it rewrites input_ids (and every
    // Parameter/subgraph derived from it, including embd) from [1, tokens] to rank-1 [tokens], and
    // requires the whole attention block (Q/K/V, KV cache, SDPA) to be shape-consistent under that
    // rewrite.
    ov::pass::Manager pass_manager;
    pass_manager.register_pass<ov::pass::SDPAToPagedAttention>();
    ASSERT_NO_THROW(pass_manager.run_passes(model));
    ASSERT_NO_THROW(model->validate_nodes_and_infer_types());

    std::shared_ptr<PagedAttentionExtension> pa;
    for (const auto& op : model->get_ordered_ops()) {
        if (auto node = ov::as_type_ptr<PagedAttentionExtension>(op)) {
            pa = node;
        }
    }
    ASSERT_NE(pa, nullptr);

    // Q (input 0) is [B_token, H*S]: its leading dim must be the flattened per-token axis, not a
    // dimension carrying head*hidden data mistakenly folded from a corrupted [tokens, tokens, ...]
    // broadcast -- i.e. its trailing dim must stay exactly hidden (H=1 head).
    const auto& q_shape = pa->get_input_partial_shape(0);
    ASSERT_TRUE(q_shape.rank().is_static());
    EXPECT_EQ(q_shape.rank().get_length(), 2);
    if (q_shape[1].is_static()) {
        EXPECT_EQ(q_shape[1].get_length(), hidden);
    }
}

// The per-layer token lookup becomes a per_layer_inputs input, so the language model takes
// embeddings only, and the unmodified PagedAttention conversion applies as for optimum-intel.
TEST(GGUFAdaptToGenAI, PagedAttentionFlattensEmbeddingsWithPerLayerInputs) {
    auto model = build_attention_gguf_model(8, 4, true);
    model->get_rt_info()["gguf_architecture"] = std::string("gemma4");
    ASSERT_TRUE(AdaptToGenAI(AdaptToGenAI::InputMode::EMBEDS_TO_LOGITS).run_on_model(model));
    ASSERT_EQ(find_parameter(model, "input_ids"), nullptr);
    ASSERT_EQ(model->input("inputs_embeds").get_partial_shape().rank().get_length(), 3);
    ASSERT_EQ(model->input("per_layer_inputs").get_partial_shape(), (ov::PartialShape{-1, -1, 2, 2}));
    ASSERT_TRUE(ov::pass::SDPAToPagedAttention().run_on_model(model));
    EXPECT_EQ(model->input("inputs_embeds").get_partial_shape(), (ov::PartialShape{-1, -1}));
    // GenAI's continuous batching supplies per_layer_inputs as [tokens, 1, layers, width].
    EXPECT_NO_THROW(model->reshape({{"inputs_embeds", {5, 4}},
                                    {"per_layer_inputs", {5, 1, 2, 2}},
                                    {"token_type_ids", {5, 1}},
                                    {"position_ids", {5}}}));
    EXPECT_NO_THROW(model->validate_nodes_and_infer_types());
}

class GGUFAdaptToGenAIEmbeddingMode : public ::testing::TestWithParam<bool> {};

TEST_P(GGUFAdaptToGenAIEmbeddingMode, MatchesTokenModeAcrossCachedDecodeAndReset) {
    const bool per_layer = GetParam();
    auto original = build_attention_gguf_model(8, 4, per_layer);
    auto embedded = original->clone();
    AdaptToGenAI().run_on_model(original);
    AdaptToGenAI adapter(AdaptToGenAI::InputMode::EMBEDS_TO_LOGITS);
    adapter.run_on_model(embedded);
    ov::Core core;
    auto tokens =
        core.compile_model(original, "CPU", ov::hint::inference_precision(ov::element::f32)).create_infer_request();
    auto values =
        core.compile_model(embedded, "CPU", ov::hint::inference_precision(ov::element::f32)).create_infer_request();
    auto lookup = core.compile_model(adapter.get_embedding_model(), "CPU").create_infer_request();
    for (int chat = 0; chat < 2; ++chat) {
        size_t past = 0;
        for (size_t length : {3, 1, 2}) {
            auto inputs = make_genai_inputs(length, past);
            lookup.set_tensor("input_ids", inputs.at("input_ids"));
            lookup.infer();
            for (const auto& entry : inputs) {
                tokens.set_tensor(entry.first, entry.second);
                if (entry.first != "input_ids")
                    values.set_tensor(entry.first, entry.second);
            }
            values.set_tensor("inputs_embeds", lookup.get_tensor("inputs_embeds"));
            if (per_layer)
                values.set_tensor("per_layer_inputs", lookup.get_tensor("per_layer_inputs"));
            tokens.infer();
            values.infer();
            auto expected = tokens.get_output_tensor(), actual = values.get_output_tensor();
            ASSERT_EQ(actual.get_shape(), expected.get_shape());
            for (size_t i = 0; i < actual.get_size(); ++i)
                EXPECT_NEAR(actual.data<float>()[i], expected.data<float>()[i], 1e-5f);
            past += length;
        }
        tokens.reset_state();
        values.reset_state();
    }
}

INSTANTIATE_TEST_SUITE_P(PerLayerInputs, GGUFAdaptToGenAIEmbeddingMode, ::testing::Bool());
