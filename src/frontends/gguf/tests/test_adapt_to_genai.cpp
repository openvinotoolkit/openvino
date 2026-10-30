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

#include <cmath>
#include <cstring>
#include <functional>
#include <memory>
#include <numeric>
#include <sstream>

#include "common_test_utils/node_builders/constant.hpp"
#include "gtest/gtest.h"
#include "model_contract.hpp"
#include "node_context.hpp"
#include "op_test_utils.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/core/visibility.hpp"
#include "openvino/frontend/gguf/adapt_to_genai.hpp"
#include "openvino/frontend/gguf/make_stateful.hpp"
#include "openvino/frontend/gguf/tokenizer_metadata.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gated_delta_net.hpp"
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
#include "openvino/op/slice.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/tanh.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/variable.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/sdpa_to_paged_attention.hpp"
#include "openvino/pass/serialize.hpp"
#include "openvino/runtime/system_conf.hpp"
#include "quant/weights.hpp"
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

// Expect 0 where `allowed(q, k)` and a large negative bias elsewhere in a [queries, keys] mask.
void expect_mask(const float* mask,
                 size_t queries,
                 size_t keys,
                 const std::function<bool(size_t, size_t)>& allowed,
                 const std::function<bool(size_t)>& skip_query = {}) {
    for (size_t q = 0; q < queries; ++q) {
        if (skip_query && skip_query(q))
            continue;
        for (size_t k = 0; k < keys; ++k) {
            if (allowed(q, k))
                EXPECT_EQ(mask[q * keys + k], 0.f) << "q=" << q << " k=" << k;
            else
                EXPECT_LT(mask[q * keys + k], -1e4f) << "q=" << q << " k=" << k;
        }
    }
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
    m.embd->get_rt_info()["gguf.token_embedding"] = true;

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

TEST(GGUFAdaptToGenAI, EmbeddingModeMovesPerLayerTokenLookupToItsOwnModel) {
    auto m = build_minimal_gguf_model(4, 2, false, true);
    m.pe_tok->get_rt_info()["gguf.per_layer_token_embedding"] = int64_t{2};
    AdaptToGenAI pass(AdaptToGenAI::InputMode::EMBEDS_TO_LOGITS);
    ASSERT_TRUE(pass.run_on_model(m.model));
    EXPECT_EQ(find_parameter(m.model, "input_ids"), nullptr);
    EXPECT_NE(find_parameter(m.model, "inputs_embeds"), nullptr);
    EXPECT_EQ(m.model->input("per_layer_inputs").get_partial_shape(), (ov::PartialShape{-1, -1, 2, 1}));
    const auto& lookup = pass.get_embedding_model();
    ASSERT_EQ(lookup->outputs().size(), 1);
    EXPECT_EQ(lookup->output(0).get_any_name(), "inputs_embeds");
    const auto& per_layer = pass.get_per_layer_embedding_model();
    ASSERT_NE(per_layer, nullptr);
    EXPECT_EQ(per_layer->get_parameters().size(), 1);
    EXPECT_EQ(per_layer->input(0).get_any_name(), "input_ids");
    ASSERT_EQ(per_layer->outputs().size(), 1);
    EXPECT_EQ(per_layer->output(0).get_any_name(), "per_layer_inputs");
    EXPECT_EQ(per_layer->output(0).get_partial_shape(), (ov::PartialShape{-1, -1, 2, 1}));
}

TEST(GGUFAdaptToGenAI, MediaPaddingAcceptsIntegralMetadataAndRejectsInvalidValues) {
    const auto adapt = [](const ov::Any& padding) {
        auto m = build_minimal_gguf_model(8, 2, false, true);
        m.pe_tok->get_rt_info()["gguf.per_layer_token_embedding"] = int64_t{2};
        auto metadata = std::make_shared<ov::frontend::gguf::GGUFTokenizerMetadata>();
        metadata->config["tokens"] =
            std::vector<std::string>{"a", "b", "pad", "<|image|>", "<|video|>", "<|audio|>", "e", "f"};
        metadata->config["padding_token_id"] = padding;
        m.model->get_rt_info()[ov::frontend::gguf::gguf_tokenizer_metadata_key()] = metadata;
        AdaptToGenAI pass(AdaptToGenAI::InputMode::EMBEDS_TO_LOGITS);
        pass.run_on_model(m.model);
        return pass.get_per_layer_embedding_model();
    };
    for (const auto& type : {ov::element::i8,
                             ov::element::u8,
                             ov::element::i16,
                             ov::element::u16,
                             ov::element::i32,
                             ov::element::u32,
                             ov::element::i64,
                             ov::element::u64}) {
        SCOPED_TRACE(type);
        const auto value = v0::Constant::create(type, {}, {2});
        auto lookup = adapt(value->get_tensor_view());
        ov::Tensor ids(ov::element::i64, {1, 4});
        std::iota(ids.data<int64_t>(), ids.data<int64_t>() + 4, 3);
        ov::TensorVector output{ov::Tensor(ov::element::f32, {1, 4, 2, 1})};
        ASSERT_TRUE(lookup->evaluate(output, {ids}));
        for (size_t i = 0; i < 3; ++i) {
            EXPECT_EQ(output.front().data<float>()[2 * i], 1004.f);
            EXPECT_EQ(output.front().data<float>()[2 * i + 1], 1005.f);
        }
        EXPECT_EQ(output.front().data<float>()[6], 1012.f);
        EXPECT_EQ(output.front().data<float>()[7], 1013.f);
    }
    for (const auto& value : {v0::Constant::create(ov::element::i64, {0}, {}),
                              v0::Constant::create(ov::element::i64, {2}, {2, 2}),
                              v0::Constant::create(ov::element::f32, {}, {2}),
                              v0::Constant::create(ov::element::i64, {}, {-1}),
                              v0::Constant::create(ov::element::u64, {}, {uint64_t(-1)}),
                              v0::Constant::create(ov::element::i64, {}, {8})}) {
        EXPECT_THROW(adapt(value->get_tensor_view()), ov::Exception);
    }
    for (const auto& value : {ov::Any{}, ov::Any{std::string{"2"}}, ov::Any{int64_t{2}}}) {
        EXPECT_THROW(adapt(value), ov::Exception);
    }
}

TEST(GGUFAdaptToGenAI, MediaPaddingPreservesTokenIdsWithoutPaddingMetadata) {
    auto m = build_minimal_gguf_model(8, 2, false, true);
    m.pe_tok->get_rt_info()["gguf.per_layer_token_embedding"] = int64_t{2};
    auto metadata = std::make_shared<ov::frontend::gguf::GGUFTokenizerMetadata>();
    metadata->config["tokens"] =
        std::vector<std::string>{"a", "b", "c", "<|image|>", "<|video|>", "<|audio|>", "e", "f"};
    m.model->get_rt_info()[ov::frontend::gguf::gguf_tokenizer_metadata_key()] = metadata;
    AdaptToGenAI pass(AdaptToGenAI::InputMode::EMBEDS_TO_LOGITS);
    ASSERT_TRUE(pass.run_on_model(m.model));
    ov::Tensor ids(ov::element::i64, {1, 4});
    std::iota(ids.data<int64_t>(), ids.data<int64_t>() + 4, 3);
    ov::TensorVector output{ov::Tensor(ov::element::f32, {1, 4, 2, 1})};
    ASSERT_TRUE(pass.get_per_layer_embedding_model()->evaluate(output, {ids}));
    for (size_t i = 0; i < output.front().get_size(); ++i)
        EXPECT_EQ(output.front().data<float>()[i], float(1006 + i));
}

// An untagged token lookup cannot be moved, so the language model keeps input_ids for it.
TEST(GGUFAdaptToGenAI, EmbeddingModePreservesUntaggedAuxiliaryTokenLookup) {
    auto m = build_minimal_gguf_model(4, 2, false, true);
    AdaptToGenAI pass(AdaptToGenAI::InputMode::EMBEDS_TO_LOGITS);
    ASSERT_TRUE(pass.run_on_model(m.model));
    EXPECT_NE(find_parameter(m.model, "input_ids"), nullptr);
    EXPECT_NE(find_parameter(m.model, "inputs_embeds"), nullptr);
    EXPECT_EQ(pass.get_embedding_model()->get_parameters().size(), 1);
}

TEST(GGUFAdaptToGenAI, EmbeddingModeRetainsScalingOnce) {
    auto m = build_minimal_gguf_model();
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
    auto swa = ov::test::utils::make_param(ov::element::f32, ov::PartialShape{1, 1, -1, -1}, "self_kq_mask_swa");
    m.model->add_parameters({swa});
    m.model->add_results({std::make_shared<v0::Result>(swa)});
    m.model->get_rt_info()[ov::frontend::gguf::pass::gguf_swa_window_key()] = int64_t{2};
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
        for (size_t layer = 0; layer < 2; ++layer) {
            const auto actual = request.get_output_tensor(layer);
            ASSERT_EQ(actual.get_shape(), (ov::Shape{2, 1, 3, past + 3}));
            for (size_t b = 0; b < 2; ++b) {
                // Padded query rows are ignored by generation.
                expect_mask(
                    actual.data<const float>() + b * 3 * (past + 3),
                    3,
                    past + 3,
                    [&](size_t q, size_t k) {
                        return k <= past + q && (b == 0 || k >= 2) && (layer == 0 || k + 2 > past + q);
                    },
                    [&](size_t q) {
                        return b == 1 && past + q < 2;
                    });
            }
        }
    }
}

using ImageAttentionMode = ov::frontend::gguf::ModelContract::ImageAttentionMode;

static void image_attention_contract(const std::shared_ptr<ov::Model>& model, ImageAttentionMode mode) {
    ov::frontend::gguf::ModelContract contract;
    contract.image_attention = mode;
    ov::frontend::gguf::write_model_contract(model, contract);
}

static std::shared_ptr<ov::Model> reload_ir_model(const std::shared_ptr<ov::Model>& model) {
    std::stringstream xml, weights;
    ov::pass::Serialize(xml, weights).run_on_model(model);
    const auto bytes = weights.str();
    ov::Tensor constants(ov::element::u8, {bytes.size()});
    if (!bytes.empty())
        std::memcpy(constants.data(), bytes.data(), bytes.size());
    return ov::Core{}.read_model(xml.str(), constants);
}

class GGUFAdaptToGenAIImageMask : public testing::TestWithParam<ImageAttentionMode> {};

TEST_P(GGUFAdaptToGenAIImageMask, RespectsLayerTypeImageGroupsAndCachedPrefix) {
    auto m = build_minimal_gguf_model();
    image_attention_contract(m.model, GetParam());
    m.model->get_rt_info()["gguf_architecture"] = std::string("custom-image-decoder");
    m.model->get_rt_info()[ov::frontend::gguf::pass::gguf_swa_window_key()] = int64_t{3};
    auto mask = find_parameter(m.model, "self_kq_mask");
    auto swa_mask = ov::test::utils::make_param(ov::element::f32, ov::PartialShape{1, 1, -1, -1}, "self_kq_mask_swa");
    m.model->add_parameters({swa_mask});
    m.model->add_results({std::make_shared<v0::Result>(mask), std::make_shared<v0::Result>(swa_mask)});
    AdaptToGenAI(AdaptToGenAI::InputMode::EMBEDS_TO_LOGITS).run_on_model(m.model);
    // Adaptation replaces the original first result with logits; the retained mask is now first.
    ov::Core core;
    for (const auto& node : m.model->get_ops())
        node->get_rt_info().clear();
    std::stringstream xml, weights;
    ov::pass::Serialize(xml, weights).run_on_model(m.model);
    const auto bytes = weights.str();
    ov::Tensor constants(ov::element::u8, {bytes.size()});
    if (!bytes.empty())
        std::memcpy(constants.data(), bytes.data(), bytes.size());
    auto restored = core.read_model(xml.str(), constants);
    for (const auto& model : {m.model, restored}) {
        auto request = core.compile_model(model, "CPU").create_infer_request();
        const std::vector<int64_t> types{0, 1, 1, 1, 1, 0, 1, 1, 0};
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
                const bool bidirectional = layer == 1 || GetParam() == ImageAttentionMode::GLOBAL_AND_SLIDING;
                SCOPED_TRACE("layer=" + std::to_string(layer) + " past=" + std::to_string(past));
                expect_mask(actual.data<const float>(), types.size(), past + types.size(), [&](size_t q, size_t k) {
                    const bool same_image = k >= past && types[q] == 1 && types[k - past] == 1 &&
                                            ((q <= 4 && k - past <= 4) || (q >= 6 && k - past >= 6));
                    return (k <= past + q && (layer == 0 || k + 3 > past + q)) || (bidirectional && same_image);
                });
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(Contract,
                         GGUFAdaptToGenAIImageMask,
                         testing::Values(ImageAttentionMode::GLOBAL_AND_SLIDING, ImageAttentionMode::SLIDING_ONLY),
                         [](const testing::TestParamInfo<ImageAttentionMode>& info) {
                             return info.param == ImageAttentionMode::GLOBAL_AND_SLIDING ? "GlobalAndSliding"
                                                                                         : "SlidingOnly";
                         });

TEST(GGUFAdaptToGenAI, EmbeddingModeMapsGenAIMultimodalPositionsToGGML) {
    auto m = build_minimal_gguf_model();
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
TEST(GGUFAdaptToGenAI, PackedTernaryEmbeddingAdaptsForBatchedLookup) {
    constexpr size_t vocab = 4;
    constexpr size_t hidden = 256;
    auto m = build_minimal_gguf_model(vocab, hidden);
    const auto bytes = load_npy<uint8_t>("pq2_0_qbytes");
    ov::Tensor payload(ov::element::u8, {bytes.size()});
    std::memcpy(payload.data(), bytes.data(), bytes.size());
    auto weight = make_weight_node(payload, "PQ2_0", {vocab, hidden});
    auto decoder = SingleOpBuilder()
                       .op("GGML_OP_GET_ROWS")
                       .input("weight", ov::element::f32, {vocab, hidden})
                       .input("inp_tokens", ov::element::i32, {1, 1, 1, -1})
                       .output("embd", ov::element::f32, {1, 1, -1, hidden})
                       .decoder();
    auto tensors = std::make_shared<TensorMap>();
    (*tensors)["weight"] = weight;
    (*tensors)["inp_tokens"] = m.inp_tokens;
    NodeContext context(decoder, tensors);
    auto embedding = op::translate_get_rows(context)[0].get_node_shared_ptr();
    embedding->set_friendly_name(m.embd->get_friendly_name());
    embedding->get_rt_info()["gguf.token_embedding"] = true;
    m.embd->output(0).replace(embedding);
    AdaptToGenAI pass(AdaptToGenAI::InputMode::EMBEDS_TO_LOGITS);
    ASSERT_TRUE(pass.run_on_model(m.model));
    const auto lookup = pass.get_embedding_model();
    ASSERT_TRUE(lookup);
    const auto reference = load_npy<float>("pq2_0_deq");
    const auto ids = make_i64_tensor({2, 3}, {3, 0, 1, 2, 1, 3});
    auto output = run_on_cpu(lookup, {{lookup->input().get_any_name(), ids}});
    ASSERT_EQ(output.get_shape(), (ov::Shape{2, 3, hidden}));
    for (size_t i = 0; i < ids.get_size(); ++i) {
        for (size_t j = 0; j < hidden; ++j)
            EXPECT_FLOAT_EQ(output.data<float>()[i * hidden + j], reference[ids.data<const int64_t>()[i] * hidden + j]);
    }
}

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

std::shared_ptr<ov::Model> build_attention_gguf_model(int64_t vocab,
                                                      int64_t hidden,
                                                      bool auxiliary_tokens = false,
                                                      ov::element::Type type = ov::element::f32) {
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
    auto vocab_table = v0::Constant::create(type, {(size_t)vocab, (size_t)hidden}, table_values);

    auto squeeze_01 = v0::Constant::create(ov::element::i64, {2}, {0, 1});
    auto axis0 = v0::Constant::create(ov::element::i64, {1}, {0});

    // Mirrors translate_get_rows's embedding-table branch exactly, same as build_minimal_gguf_model.
    auto indices = std::make_shared<v0::Squeeze>(inp_tokens, squeeze_01);
    auto gather = std::make_shared<v8::Gather>(vocab_table, indices, axis0);
    auto embd = std::make_shared<v0::Unsqueeze>(gather, axis0);
    embd->set_friendly_name("embd");
    embd->get_rt_info()["gguf.token_embedding"] = true;

    auto squeeze_axis1 = v0::Constant::create(ov::element::i64, {1}, {1});
    ov::Output<ov::Node> combined = embd;
    if (auxiliary_tokens) {
        auto auxiliary = std::make_shared<v8::Gather>(vocab_table, indices, axis0);
        auto lifted = std::make_shared<v0::Unsqueeze>(auxiliary, axis0);
        lifted->set_friendly_name("per_layer_tokens");
        // Two layers of hidden / 2 values each, as Gemma4's per-layer token embedding.
        lifted->get_rt_info()["gguf.per_layer_token_embedding"] = int64_t{2};
        combined = std::make_shared<v1::Add>(embd, lifted);
    }
    auto embd_3d = std::make_shared<v0::Squeeze>(std::make_shared<v0::Convert>(combined, type), squeeze_axis1);

    std::vector<float> zero_w(hidden * hidden, 0.0f);
    auto w_zero = v0::Constant::create(type, {(size_t)hidden, (size_t)hidden}, zero_w);
    std::vector<float> identity_w(hidden * hidden, 0.0f);
    for (int64_t i = 0; i < hidden; ++i) {
        identity_w[i * hidden + i] = 1.0f;
    }
    auto w_identity = v0::Constant::create(type, {(size_t)hidden, (size_t)hidden}, identity_w);

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
        auto var =
            std::make_shared<util::Variable>(util::VariableInfo{ov::PartialShape{-1, 1, -1, hidden}, type, var_id});
        auto init_shape = v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{1, 1, 0, hidden});
        auto init = std::make_shared<v3::Broadcast>(v0::Constant::create(type, {}, {0.0f}), init_shape);
        auto read = std::make_shared<v6::ReadValue>(init, var);
        auto past = std::make_shared<v8::Gather>(read, beam_idx, axis0, 0);
        auto concat = std::make_shared<v0::Concat>(ov::OutputVector{past, cur}, -2);
        auto assign = std::make_shared<v6::Assign>(concat, var);
        return std::make_pair(ov::Output<ov::Node>(concat), std::static_pointer_cast<Sink>(assign));
    };
    auto [k_concat, k_assign] = make_kv_cache(k, "attn_k_cache.0");
    auto [v_concat, v_assign] = make_kv_cache(v, "attn_v_cache.0");

    auto scale = v0::Constant::create(type, {}, {1.0f});
    auto sdpa = std::make_shared<v13::ScaledDotProductAttention>(q,
                                                                 k_concat,
                                                                 v_concat,
                                                                 std::make_shared<v0::Convert>(self_kq_mask, type),
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

class GGUFAdaptToGenAISlidingWindow : public testing::TestWithParam<std::tuple<bool, ov::element::Type, bool>> {};

TEST_P(GGUFAdaptToGenAISlidingWindow, PreservedByPagedAttention) {
    const auto& [embeds, mask_type, sliced] = GetParam();
    auto model = build_attention_gguf_model(16, 4, false, mask_type);
    auto mask = find_parameter(model, "self_kq_mask");
    auto swa = ov::test::utils::make_param(ov::element::f32, ov::PartialShape{1, 1, -1, -1}, "self_kq_mask_swa");
    model->add_parameters({swa});
    ov::Output<ov::Node> mask_source = swa;
    if (sliced) {
        mask_source = std::make_shared<v8::Slice>(swa,
                                                  v0::Constant::create(ov::element::i64, {1}, {0}),
                                                  v0::Constant::create(ov::element::i64, {1}, {1}),
                                                  v0::Constant::create(ov::element::i64, {1}, {1}),
                                                  v0::Constant::create(ov::element::i64, {1}, {1}));
    }
    mask->output(0).replace(mask_source);
    image_attention_contract(model, ImageAttentionMode::SLIDING_ONLY);
    model->get_rt_info()[ov::frontend::gguf::pass::gguf_swa_window_key()] = int64_t{64};
    AdaptToGenAI(embeds ? AdaptToGenAI::InputMode::EMBEDS_TO_LOGITS : AdaptToGenAI::InputMode::IDS_TO_LOGITS)
        .run_on_model(model);
    ov::pass::SDPAToPagedAttention(false, false, false).run_on_model(model);
    size_t count = 0;
    for (const auto& node : model->get_ordered_ops()) {
        if (const auto pa = ov::as_type_ptr<ov::op::PagedAttentionExtension>(node)) {
            const auto window = ov::util::get_constant_from_source(pa->input_value(10));
            ASSERT_NE(window, nullptr);
            EXPECT_EQ(window->cast_vector<int32_t>(), std::vector<int32_t>{64});
            ++count;
        }
    }
    EXPECT_EQ(count, 1);
    for (const auto& node : model->get_ops()) {
        if (const auto pa = ov::as_type_ptr<ov::op::PagedAttentionExtension>(node)) {
            auto window_model =
                std::make_shared<ov::Model>(ov::OutputVector{pa->input_value(10)}, ov::ParameterVector{});
            ov::pass::ConstantFolding().run_on_model(window_model);
            const auto window =
                ov::as_type_ptr<v0::Constant>(window_model->get_results()[0]->get_input_node_shared_ptr(0));
            ASSERT_NE(window, nullptr);
            EXPECT_EQ(window->cast_vector<int32_t>(), std::vector<int32_t>{64});
        }
    }
}

class GGUFAdaptToGenAIImageWindow : public testing::TestWithParam<bool> {};

TEST_P(GGUFAdaptToGenAIImageWindow, MatchesSDPAAndPagedAttention) {
    const bool paged = GetParam();
#if defined(OPENVINO_ARCH_ARM64)
    if (paged && !ov::with_cpu_sve())
        GTEST_SKIP() << "CPU PagedAttention requires SVE on ARM64";
#elif !defined(OPENVINO_ARCH_X86_64)
    if (paged)
        GTEST_SKIP() << "CPU PagedAttention requires x86-64 or ARM64 with SVE";
#endif
    const size_t hidden = 4;
    const std::vector<int32_t> types{0, 1, 1, 1, 1, 0, 1, 1, 0};
    const auto length = types.size();
    ov::Core core;
    auto model = build_attention_gguf_model(16, hidden);
    auto mask = find_parameter(model, "self_kq_mask");
    auto swa = ov::test::utils::make_param(ov::element::f32, ov::PartialShape{1, 1, -1, -1}, "self_kq_mask_swa");
    model->add_parameters({swa});
    mask->output(0).replace(swa->output(0));
    image_attention_contract(model, ImageAttentionMode::SLIDING_ONLY);
    model->get_rt_info()[ov::frontend::gguf::pass::gguf_swa_window_key()] = int64_t{3};
    AdaptToGenAI(AdaptToGenAI::InputMode::EMBEDS_TO_LOGITS).run_on_model(model);
    if (paged)
        ov::pass::SDPAToPagedAttention(false, false, false).run_on_model(model);
    auto compiled = core.compile_model(model,
                                       "CPU",
                                       ov::hint::inference_precision(ov::element::f32),
                                       ov::hint::kv_cache_precision(ov::element::f32));
    auto request = compiled.create_infer_request();
    for (const auto& input : compiled.inputs()) {
        const auto name = input.get_any_name();
        if (name == "inputs_embeds") {
            ov::Tensor tensor(ov::element::f32, paged ? ov::Shape{length, hidden} : ov::Shape{1, length, hidden});
            for (size_t i = 0; i < tensor.get_size(); ++i)
                tensor.data<float>()[i] = static_cast<float>(i / hidden);
            request.set_tensor(input, tensor);
        } else if (name == "token_type_ids" || name == "position_ids") {
            ov::Tensor tensor(input.get_element_type(),
                              input.get_partial_shape().rank() == 1 ? ov::Shape{length} : ov::Shape{1, length});
            for (size_t i = 0; i < length; ++i) {
                const auto value = name == "token_type_ids" ? types[i] : static_cast<int32_t>(i);
                if (tensor.get_element_type() == ov::element::i32)
                    tensor.data<int32_t>()[i] = value;
                else
                    tensor.data<int64_t>()[i] = value;
            }
            request.set_tensor(input, tensor);
        } else if (name == "attention_mask") {
            ov::Tensor tensor(ov::element::i64, {1, length});
            std::fill_n(tensor.data<int64_t>(), length, 1);
            request.set_tensor(input, tensor);
        } else if (name.find("key_cache.") == 0 || name.find("value_cache.") == 0) {
            auto shape = input.get_partial_shape();
            shape[0] = 1;
            request.set_tensor(input, ov::Tensor(input.get_element_type(), shape.get_shape()));
        } else {
            std::vector<int32_t> values;
            if (name == "beam_idx" || name == "past_lens" || name == "block_indices")
                values = {0};
            else if (name == "subsequence_begins")
                values = {0, static_cast<int32_t>(length)};
            else if (name == "block_indices_begins")
                values = {0, 1};
            else if (name == "max_context_len")
                values = {static_cast<int32_t>(length)};
            else
                FAIL() << "Unexpected input: " << name;
            const auto shape = input.get_partial_shape().rank() == 0 ? ov::Shape{} : ov::Shape{values.size()};
            ov::Tensor tensor(ov::element::i32, shape);
            std::copy(values.begin(), values.end(), tensor.data<int32_t>());
            request.set_tensor(input, tensor);
        }
    }
    request.infer();
    const auto output = request.get_output_tensor(0);
    ASSERT_EQ(output.get_size(), length * hidden);
    for (size_t q = 0; q < length; ++q) {
        float sum = 0;
        size_t count = 0;
        for (size_t k = 0; k < length; ++k) {
            const bool same_image = types[q] == 1 && types[k] == 1 && ((q <= 4 && k <= 4) || (q >= 6 && k >= 6));
            if ((k <= q && k + 3 > q) || same_image) {
                sum += static_cast<float>(k);
                ++count;
            }
        }
        for (size_t h = 0; h < hidden; ++h)
            EXPECT_NEAR(output.data<const float>()[q * hidden + h], sum / count, 1e-5f);
    }
}

INSTANTIATE_TEST_SUITE_P(SDPAAndPagedAttention,
                         GGUFAdaptToGenAIImageWindow,
                         testing::Bool(),
                         [](const testing::TestParamInfo<bool>& info) {
                             return info.param ? "PagedAttention" : "SDPA";
                         });

INSTANTIATE_TEST_SUITE_P(GGUFAdaptToGenAI,
                         GGUFAdaptToGenAISlidingWindow,
                         testing::Combine(testing::Bool(),
                                          testing::Values(ov::element::f32, ov::element::f16, ov::element::bf16),
                                          testing::Bool()));

constexpr float lm_head_soft_cap = 5.0f;

float lm_head_weight(int64_t row, int64_t col) {
    return 0.25f * static_cast<float>(row - col);
}

// Replaces the logits Result's input with the LM head as the frontend emits it: a rank-4
// [1, 1, tokens, hidden] MatMul against [vocab, hidden], optionally followed by Gemma's final-logit
// softcap (x * (1/cap) -> tanh -> * cap) with [1, 1, 1, 1] scale constants.
void attach_lm_head(const std::shared_ptr<ov::Model>& model,
                    const ov::Output<ov::Node>& hidden_states,
                    int64_t vocab,
                    int64_t hidden,
                    bool softcap) {
    ov::Output<ov::Node> hidden_4d = hidden_states;
    if (hidden_states.get_partial_shape().size() == 3) {
        hidden_4d = std::make_shared<v0::Unsqueeze>(hidden_states, v0::Constant::create(ov::element::i64, {1}, {0}));
    }
    std::vector<float> weight(vocab * hidden);
    for (int64_t v = 0; v < vocab; ++v) {
        for (int64_t h = 0; h < hidden; ++h) {
            weight[v * hidden + h] = lm_head_weight(v, h);
        }
    }
    ov::Output<ov::Node> logits =
        std::make_shared<v0::MatMul>(hidden_4d,
                                     v0::Constant::create(ov::element::f32, {(size_t)vocab, (size_t)hidden}, weight),
                                     false,
                                     true);
    if (softcap) {
        auto scale_down = v0::Constant::create(ov::element::f32, {1, 1, 1, 1}, {1.0f / lm_head_soft_cap});
        auto scale_up = v0::Constant::create(ov::element::f32, {1, 1, 1, 1}, {lm_head_soft_cap});
        logits = std::make_shared<v1::Multiply>(logits, scale_down);
        logits = std::make_shared<v0::Tanh>(logits);
        logits = std::make_shared<v1::Multiply>(logits, scale_up);
    }
    model->get_results()[0]->input(0).replace_source_output(logits);
    model->validate_nodes_and_infer_types();
}

// GenAI's gather/slice-before-matmul only finds the LM head as Result <- MatMul, optionally behind
// Divide -> Tanh -> Multiply, with a rank-3 MatMul input; returns the MatMul when that holds.
std::shared_ptr<v0::MatMul> find_genai_lm_head(const std::shared_ptr<ov::Model>& model, bool softcap) {
    auto node = model->get_results()[0]->get_input_node_shared_ptr(0);
    if (softcap) {
        const auto scale_up = ov::as_type_ptr<v1::Multiply>(node);
        if (!scale_up) {
            return nullptr;
        }
        const auto tanh = ov::as_type_ptr<v0::Tanh>(scale_up->get_input_node_shared_ptr(0));
        if (!tanh) {
            return nullptr;
        }
        const auto scale_down = ov::as_type_ptr<v1::Divide>(tanh->get_input_node_shared_ptr(0));
        if (!scale_down) {
            return nullptr;
        }
        node = scale_down->get_input_node_shared_ptr(0);
    }
    auto matmul = ov::as_type_ptr<v0::MatMul>(node);
    if (!matmul || matmul->get_input_partial_shape(0).size() != 3) {
        return nullptr;
    }
    return matmul;
}

}  // namespace

TEST(GGUFAdaptToGenAI, PackedTernaryEmbeddingSurvivesPagedAttentionPrefill) {
    constexpr size_t vocab = 4;
    constexpr size_t hidden = 256;
    auto model = build_attention_gguf_model(vocab, hidden);
    const auto bytes = load_npy<uint8_t>("pq2_0_qbytes");
    ov::Tensor payload(ov::element::u8, {bytes.size()});
    std::memcpy(payload.data(), bytes.data(), bytes.size());
    auto weight = make_weight_node(payload, "PQ2_0", {vocab, hidden});
    std::shared_ptr<ov::Node> old_embedding;
    for (const auto& node : model->get_ops()) {
        if (node->get_friendly_name() == "embd")
            old_embedding = node;
    }
    ASSERT_TRUE(old_embedding);
    const auto indices = old_embedding->input_value(0).get_node_shared_ptr()->input_value(1);
    auto embedding = std::make_shared<v0::Unsqueeze>(gather_weight_rows(weight, indices),
                                                     v0::Constant::create(ov::element::i64, {1}, {0}));
    embedding->set_friendly_name("embd");
    embedding->get_rt_info()["gguf.token_embedding"] = true;
    old_embedding->output(0).replace(embedding);
    ASSERT_TRUE(AdaptToGenAI().run_on_model(model));
    ASSERT_TRUE(ov::pass::SDPAToPagedAttention().run_on_model(model));
    auto ids = find_parameter(model, "input_ids");
    ASSERT_TRUE(ids);
    ids->set_partial_shape({3});
    ASSERT_NO_THROW(model->validate_nodes_and_infer_types());
    size_t attention_count = 0;
    for (const auto& node : model->get_ops()) {
        if (auto attention = ov::as_type_ptr<PagedAttentionExtension>(node)) {
            ++attention_count;
            EXPECT_EQ(attention->get_input_partial_shape(0), (ov::PartialShape{3, hidden}));
        }
    }
    EXPECT_EQ(attention_count, 1);
}

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

class GGUFAdaptToGenAILmHead : public ::testing::TestWithParam<bool> {};

// Without the rank-3 Result <- MatMul form, GenAI can't select the sampled tokens before the LM
// head, and PagedAttention prefill projects every prompt token to vocab.
TEST_P(GGUFAdaptToGenAILmHead, ExposesHeadForGenAITokenSelection) {
    const bool softcap = GetParam();
    const int64_t vocab = 6, hidden = 3;
    auto m = build_minimal_gguf_model(vocab, hidden);
    attach_lm_head(m.model, m.embd, vocab, hidden, softcap);

    ASSERT_TRUE(AdaptToGenAI().run_on_model(m.model));
    ASSERT_NO_THROW(m.model->validate_nodes_and_infer_types());
    ASSERT_EQ(m.model->get_results().size(), 1);
    ASSERT_NE(find_genai_lm_head(m.model, softcap), nullptr);
    if (softcap) {
        const auto tanh = m.model->get_results()[0]->get_input_node_shared_ptr(0)->get_input_node_shared_ptr(0);
        const auto divisor = ov::as_type_ptr<v0::Constant>(tanh->get_input_node_ptr(0)->get_input_node_shared_ptr(1));
        ASSERT_NE(divisor, nullptr);
        EXPECT_FLOAT_EQ(divisor->cast_vector<float>()[0], lm_head_soft_cap);
    }

    ov::Core core;
    auto request =
        core.compile_model(m.model, "CPU", ov::hint::inference_precision(ov::element::f32)).create_infer_request();
    const size_t length = 3;
    for (const auto& [name, tensor] : make_genai_inputs(length)) {
        request.set_tensor(name, tensor);
    }
    request.infer();
    const auto logits = request.get_tensor("logits");
    ASSERT_EQ(logits.get_shape(), (ov::Shape{1, length, (size_t)vocab}));
    for (size_t token = 0; token < length; ++token) {
        for (int64_t v = 0; v < vocab; ++v) {
            float expected = 0.0f;
            for (int64_t h = 0; h < hidden; ++h) {
                expected += lm_head_weight(v, h) * static_cast<float>(token * hidden + h);
            }
            if (softcap) {
                expected = lm_head_soft_cap * std::tanh(expected / lm_head_soft_cap);
            }
            EXPECT_NEAR(logits.data<float>()[token * vocab + v], expected, 1e-4f);
        }
    }
}

// GenAI applies its token selection after SDPAToPagedAttention, so the head must keep its form there.
TEST_P(GGUFAdaptToGenAILmHead, KeepsHeadFormThroughSDPAToPagedAttention) {
    const bool softcap = GetParam();
    const int64_t vocab = 8, hidden = 4;
    auto model = build_attention_gguf_model(vocab, hidden);
    attach_lm_head(model, model->get_results()[0]->input_value(0), vocab, hidden, softcap);

    ASSERT_TRUE(AdaptToGenAI().run_on_model(model));
    ASSERT_TRUE(ov::pass::SDPAToPagedAttention().run_on_model(model));
    ASSERT_NO_THROW(model->validate_nodes_and_infer_types());
    EXPECT_NE(find_genai_lm_head(model, softcap), nullptr);
}

INSTANTIATE_TEST_SUITE_P(Softcap, GGUFAdaptToGenAILmHead, ::testing::Bool());

// The per-layer token lookup becomes a per_layer_inputs input, so the language model takes
// embeddings only, and the unmodified PagedAttention conversion applies as for optimum-intel.
TEST(GGUFAdaptToGenAI, PagedAttentionFlattensEmbeddingsWithPerLayerInputs) {
    auto model = build_attention_gguf_model(8, 1536, true);
    image_attention_contract(model, ImageAttentionMode::CAUSAL);
    ASSERT_TRUE(AdaptToGenAI(AdaptToGenAI::InputMode::EMBEDS_TO_LOGITS).run_on_model(model));
    ASSERT_EQ(find_parameter(model, "input_ids"), nullptr);
    ASSERT_EQ(model->input("inputs_embeds").get_partial_shape().rank().get_length(), 3);
    ASSERT_EQ(model->input("per_layer_inputs").get_partial_shape(), (ov::PartialShape{-1, -1, 2, 768}));
    // llama.cpp keeps E2B/E4B image attention causal.
    ASSERT_EQ(find_parameter(model, "token_type_ids"), nullptr);
    ASSERT_TRUE(ov::pass::SDPAToPagedAttention().run_on_model(model));
    EXPECT_EQ(model->input("inputs_embeds").get_partial_shape(), (ov::PartialShape{-1, -1}));
    // GenAI's continuous batching supplies per_layer_inputs as [tokens, 1, layers, width].
    EXPECT_NO_THROW(
        model->reshape({{"inputs_embeds", {5, 1536}}, {"per_layer_inputs", {5, 1, 2, 768}}, {"position_ids", {5}}}));
    EXPECT_NO_THROW(model->validate_nodes_and_infer_types());
}

class GGUFAdaptToGenAIImageAttentionContract
    : public ::testing::TestWithParam<std::tuple<int64_t, ImageAttentionMode>> {};

TEST_P(GGUFAdaptToGenAIImageAttentionContract, UsesDeclaredSemanticsForAnyEmbeddingWidth) {
    const auto& [hidden, mode] = GetParam();
    auto model = build_attention_gguf_model(8, hidden, true);
    image_attention_contract(model, mode);
    model->get_rt_info()["gguf_architecture"] = std::string("custom-image-decoder");
    ASSERT_TRUE(AdaptToGenAI(AdaptToGenAI::InputMode::EMBEDS_TO_LOGITS).run_on_model(model));
    EXPECT_EQ(find_parameter(model, "token_type_ids") == nullptr, mode == ImageAttentionMode::CAUSAL);
}

INSTANTIATE_TEST_SUITE_P(EmbeddingWidth,
                         GGUFAdaptToGenAIImageAttentionContract,
                         ::testing::Combine(::testing::Values(int64_t{32}, int64_t{1536}, int64_t{2560}),
                                            ::testing::Values(ImageAttentionMode::CAUSAL,
                                                              ImageAttentionMode::GLOBAL_AND_SLIDING,
                                                              ImageAttentionMode::SLIDING_ONLY)));

TEST(GGUFAdaptToGenAI, ModelContractSurvivesIRReloadBeforeAdaptation) {
    for (const auto batchable : std::vector<std::optional<bool>>{std::nullopt, false, true}) {
        auto model = build_attention_gguf_model(8, 4);
        ov::frontend::gguf::ModelContract contract;
        contract.image_attention = ImageAttentionMode::SLIDING_ONLY;
        contract.batchable_recurrent = batchable;
        ov::frontend::gguf::write_model_contract(model, contract);
        for (const auto& node : model->get_ops())
            node->get_rt_info().clear();
        std::stringstream xml, weights;
        ov::pass::Serialize(xml, weights).run_on_model(model);
        const auto bytes = weights.str();
        ov::Tensor constants(ov::element::u8, {bytes.size()});
        if (!bytes.empty())
            std::memcpy(constants.data(), bytes.data(), bytes.size());
        auto restored = ov::Core{}.read_model(xml.str(), constants);
        const auto restored_contract = ov::frontend::gguf::read_model_contract(restored);
        EXPECT_EQ(restored_contract.image_attention, ImageAttentionMode::SLIDING_ONLY);
        EXPECT_EQ(restored_contract.batchable_recurrent, batchable);
        ASSERT_TRUE(AdaptToGenAI(AdaptToGenAI::InputMode::IDS_TO_LOGITS).run_on_model(restored));
        EXPECT_NE(find_parameter(restored, "input_ids"), nullptr);
    }
}

TEST(GGUFAdaptToGenAI, LegacyRecurrentContractSurvivesIRReloadBeforeAdaptation) {
    for (const std::string architecture : {"lfm2", "lfm2moe", "qwen35", "custom-recurrent-decoder"}) {
        SCOPED_TRACE(architecture);
        auto model = build_minimal_gguf_model().model;
        model->get_rt_info()["gguf_architecture"] = architecture;
        model->get_rt_info()[ov::frontend::gguf::pass::gguf_recurrent_states_key()] = std::vector<std::string>{};
        auto restored = reload_ir_model(model);
        ASSERT_FALSE(restored->has_rt_info({"gguf_model_contract"}));
        ASSERT_EQ(restored->get_rt_info<std::string>({"gguf_architecture"}), architecture);
        const bool batchable = architecture == "lfm2" || architecture == "lfm2moe";
        EXPECT_EQ(ov::frontend::gguf::read_model_contract(restored).batchable_recurrent,
                  batchable ? std::optional<bool>{true} : std::nullopt);
        ASSERT_TRUE(AdaptToGenAI().run_on_model(restored));
        EXPECT_EQ(restored->input("input_ids").get_partial_shape()[0],
                  batchable ? ov::Dimension::dynamic() : ov::Dimension{1});
    }
}

class GGUFAdaptToGenAILegacyImageContract
    : public testing::TestWithParam<std::tuple<std::string, int64_t, ImageAttentionMode>> {};

TEST_P(GGUFAdaptToGenAILegacyImageContract, ReloadsOldSchemaAndKeepsGlobalAndSlidingMasks) {
    const auto& [architecture, hidden, mode] = GetParam();
    auto m = build_minimal_gguf_model(4, hidden);
    const auto lookup = m.embd->get_input_node_shared_ptr(0);
    const auto table = lookup->get_input_node_shared_ptr(0);
    table->set_friendly_name("token_embd.weight");
    m.model->get_rt_info()["gguf_architecture"] = architecture;
    m.model->get_rt_info()[ov::frontend::gguf::pass::gguf_swa_window_key()] = int64_t{1};
    auto global_mask = find_parameter(m.model, "self_kq_mask");
    auto sliding_mask =
        ov::test::utils::make_param(ov::element::f32, ov::PartialShape{1, 1, -1, -1}, "self_kq_mask_swa");
    auto global_result = std::make_shared<v0::Result>(global_mask);
    auto sliding_result = std::make_shared<v0::Result>(sliding_mask);
    m.model->add_parameters({sliding_mask});
    m.model->add_results({global_result, sliding_result});
    auto restored = reload_ir_model(m.model);
    ASSERT_FALSE(restored->has_rt_info({"gguf_model_contract"}));
    ASSERT_EQ(restored->get_rt_info<std::string>({"gguf_architecture"}), architecture);
    for (const auto& node : restored->get_ops())
        ASSERT_EQ(node->get_rt_info().count("gguf.token_embedding"), 0);
    EXPECT_EQ(ov::frontend::gguf::read_model_contract(restored, hidden).image_attention, mode);
    AdaptToGenAI adapter(AdaptToGenAI::InputMode::EMBEDS_TO_LOGITS);
    ASSERT_TRUE(adapter.run_on_model(restored));
    ASSERT_NE(adapter.get_embedding_model(), nullptr);
    ASSERT_EQ(find_parameter(restored, "input_ids"), nullptr);
    EXPECT_EQ(restored->input("inputs_embeds").get_partial_shape(), (ov::PartialShape{-1, -1, hidden}));
    EXPECT_EQ(find_parameter(restored, "token_type_ids") == nullptr, mode == ImageAttentionMode::CAUSAL);
    auto inputs = make_genai_inputs(3);
    inputs.erase("input_ids");
    ov::Tensor embeddings(ov::element::f32, {1, 3, static_cast<size_t>(hidden)});
    std::fill_n(embeddings.data<float>(), embeddings.get_size(), 0.f);
    inputs.emplace("inputs_embeds", embeddings);
    if (mode != ImageAttentionMode::CAUSAL) {
        ov::Tensor token_types(ov::element::i64, {1, 3});
        const std::vector<int64_t> types{1, 1, 0};
        std::copy(types.begin(), types.end(), token_types.data<int64_t>());
        inputs.emplace("token_type_ids", token_types);
    }
    ov::Core core;
    auto request = core.compile_model(restored, "CPU").create_infer_request();
    for (const auto& [name, tensor] : inputs)
        request.set_tensor(name, tensor);
    request.infer();
    for (size_t layer = 0; layer < 2; ++layer) {
        const auto mask = request.get_output_tensor(layer);
        ASSERT_EQ(mask.get_shape(), (ov::Shape{1, 1, 3, 3}));
        expect_mask(mask.data<const float>(), 3, 3, [&, mode = mode](size_t q, size_t k) {
            const bool image_attention = mode == ImageAttentionMode::GLOBAL_AND_SLIDING ||
                                         (layer == 1 && mode == ImageAttentionMode::SLIDING_ONLY);
            return (k <= q && (layer == 0 || k + 1 > q)) || (image_attention && q < 2 && k < 2);
        });
    }
}

INSTANTIATE_TEST_SUITE_P(
    ArchitectureAndWidth,
    GGUFAdaptToGenAILegacyImageContract,
    testing::Values(std::make_tuple(std::string{"gemma3"}, int64_t{32}, ImageAttentionMode::GLOBAL_AND_SLIDING),
                    std::make_tuple(std::string{"gemma3"}, int64_t{1536}, ImageAttentionMode::GLOBAL_AND_SLIDING),
                    std::make_tuple(std::string{"gemma3"}, int64_t{2560}, ImageAttentionMode::GLOBAL_AND_SLIDING),
                    std::make_tuple(std::string{"gemma4"}, int64_t{32}, ImageAttentionMode::SLIDING_ONLY),
                    std::make_tuple(std::string{"gemma4"}, int64_t{1536}, ImageAttentionMode::CAUSAL),
                    std::make_tuple(std::string{"gemma4"}, int64_t{2560}, ImageAttentionMode::CAUSAL),
                    std::make_tuple(std::string{"gemma4"}, int64_t{3072}, ImageAttentionMode::SLIDING_ONLY)));

TEST(GGUFAdaptToGenAI, ExplicitPartialContractPreventsLegacyRecurrentFallbackAfterIRReload) {
    const std::vector<ov::AnyMap> contracts{ov::AnyMap{},
                                            ov::AnyMap{{"batchable_recurrent", std::string{"auto"}}},
                                            ov::AnyMap{{"batchable_recurrent", std::string{"false"}}},
                                            ov::AnyMap{{"image_attention", std::string{"global_and_sliding"}}}};
    for (const std::string architecture : {"lfm2", "lfm2moe"}) {
        for (const auto& contract : contracts) {
            SCOPED_TRACE(architecture);
            auto model = build_minimal_gguf_model().model;
            model->get_rt_info()["gguf_architecture"] = architecture;
            model->get_rt_info()["gguf_model_contract"] = contract;
            model->get_rt_info()[ov::frontend::gguf::pass::gguf_recurrent_states_key()] = std::vector<std::string>{};
            auto restored = reload_ir_model(model);
            ASSERT_TRUE(restored->has_rt_info({"gguf_model_contract"}));
            EXPECT_NE(ov::frontend::gguf::read_model_contract(restored).batchable_recurrent, std::optional<bool>{true});
            ASSERT_TRUE(AdaptToGenAI().run_on_model(restored));
            EXPECT_EQ(restored->input("input_ids").get_partial_shape()[0], ov::Dimension{1});
        }
    }
}

TEST(GGUFAdaptToGenAI, ExplicitPartialContractPreventsLegacyImageFallbackAfterIRReload) {
    const std::vector<ov::AnyMap> contracts{ov::AnyMap{},
                                            ov::AnyMap{{"batchable_recurrent", std::string{"true"}}},
                                            ov::AnyMap{{"image_attention", std::string{"causal"}}}};
    for (const std::string architecture : {"gemma3", "gemma4"}) {
        for (const auto& contract : contracts) {
            SCOPED_TRACE(architecture);
            auto m = build_minimal_gguf_model(4, 32);
            const auto lookup = m.embd->get_input_node_shared_ptr(0);
            const auto table = lookup->get_input_node_shared_ptr(0);
            table->set_friendly_name("token_embd.weight");
            m.model->get_rt_info()["gguf_architecture"] = architecture;
            m.model->get_rt_info()["gguf_model_contract"] = contract;
            auto restored = reload_ir_model(m.model);
            ASSERT_TRUE(restored->has_rt_info({"gguf_model_contract"}));
            EXPECT_EQ(ov::frontend::gguf::read_model_contract(restored, 32).image_attention,
                      ImageAttentionMode::CAUSAL);
            ASSERT_TRUE(AdaptToGenAI(AdaptToGenAI::InputMode::EMBEDS_TO_LOGITS).run_on_model(restored));
            EXPECT_EQ(find_parameter(restored, "token_type_ids"), nullptr);
        }
    }
}

static MinimalGgufModel build_scaled_legacy_gguf_model(const std::string& architecture, bool per_layer) {
    auto m = build_minimal_gguf_model(4, 4);
    const auto squeeze_axes = v0::Constant::create(ov::element::i64, {2}, {0, 1});
    const auto lookup = m.embd->get_input_node_shared_ptr(0);
    const auto original_table = ov::as_type_ptr<v0::Constant>(lookup->get_input_node_shared_ptr(0));
    const auto values = original_table->cast_vector<float>();
    const auto table = v0::Constant::create(ov::element::f32, {1, 1, 4, 4}, values);
    table->set_friendly_name("Constant_legacy_token_embd.weight");
    const auto squeezed_table = std::make_shared<v0::Squeeze>(table, squeeze_axes);
    lookup->input(0).replace_source_output(squeezed_table);
    m.embd->set_friendly_name("Unsqueeze_legacy_lookup");
    m.embd->get_rt_info().erase("gguf.token_embedding");
    const auto converted = std::make_shared<v0::Convert>(m.embd, ov::element::f32);
    converted->set_friendly_name("Convert_legacy_embd");
    const auto scale = v0::Constant::create(ov::element::f32, {}, {2.f});
    const auto scaled = std::make_shared<v1::Multiply>(converted, scale);
    scaled->set_friendly_name("Multiply_legacy_embd_scaled");
    scaled->get_rt_info()["gguf.token_embedding"] = true;
    const auto logits = m.model->get_results().front()->get_input_node_shared_ptr(0);
    logits->input(0).replace_source_output(scaled);
    if (per_layer) {
        std::vector<float> per_layer_values(24);
        std::iota(per_layer_values.begin(), per_layer_values.end(), 100.f);
        const auto pe_table = v0::Constant::create(ov::element::f32, {1, 1, 4, 6}, per_layer_values);
        pe_table->set_friendly_name("Constant_legacy_per_layer_token_embd.weight");
        const auto pe_table_squeezed = std::make_shared<v0::Squeeze>(pe_table, squeeze_axes);
        const auto indices = std::make_shared<v0::Squeeze>(m.inp_tokens, squeeze_axes);
        const auto axis0 = v0::Constant::create(ov::element::i64, {1}, {0});
        const auto gathered = std::make_shared<v8::Gather>(pe_table_squeezed, indices, axis0);
        m.pe_tok = std::make_shared<v0::Unsqueeze>(gathered, axis0);
        m.pe_tok->set_friendly_name("Unsqueeze_legacy_pe_tok_lookup");
        const auto pe_converted = std::make_shared<v0::Convert>(m.pe_tok, ov::element::f32);
        pe_converted->set_friendly_name("Convert_legacy_pe_tok_flat");
        const auto per_layer_scale = std::sqrt(2.f);
        const auto pe_scale = v0::Constant::create(ov::element::f32, {}, {per_layer_scale});
        const auto pe_scaled = std::make_shared<v1::Multiply>(pe_converted, pe_scale);
        pe_scaled->set_friendly_name("Multiply_legacy_pe_tok_flat_scaled");
        pe_scaled->get_rt_info()["gguf.per_layer_token_embedding"] = int64_t{3};
        const auto split_shape = v0::Constant::create(ov::element::i64, {4}, {0, -1, 3, 2});
        const auto split = std::make_shared<v1::Reshape>(pe_scaled, split_shape, true);
        split->set_friendly_name("Reshape_legacy_pe_tok_split");
        const auto transpose_axes = v0::Constant::create(ov::element::i64, {4}, {0, 2, 1, 3});
        const auto layer_major = std::make_shared<v1::Transpose>(split, transpose_axes);
        layer_major->set_friendly_name("Transpose_legacy_pe_tok");
        const auto reduce_axes = v0::Constant::create(ov::element::i64, {2}, {1, 3});
        const auto pe_sum = std::make_shared<v1::ReduceSum>(layer_major, reduce_axes, true);
        const auto combined = std::make_shared<v1::Add>(logits, pe_sum);
        m.model->get_results().front()->input(0).replace_source_output(combined);
    }
    m.model->get_rt_info()["gguf_architecture"] = architecture;
    m.model->validate_nodes_and_infer_types();
    return m;
}

class GGUFAdaptToGenAILegacyScaledEmbedding : public testing::TestWithParam<std::pair<std::string, bool>> {};

TEST_P(GGUFAdaptToGenAILegacyScaledEmbedding, ReloadPreservesTokenScalingAndInjectedMediaValues) {
    const auto& [architecture, per_layer] = GetParam();
    auto m = build_scaled_legacy_gguf_model(architecture, per_layer);
    auto restored = reload_ir_model(m.model);
    ASSERT_FALSE(restored->has_rt_info({"gguf_model_contract"}));
    for (const auto& node : restored->get_ops()) {
        ASSERT_EQ(node->get_rt_info().count("gguf.token_embedding"), 0);
        ASSERT_EQ(node->get_rt_info().count("gguf.per_layer_token_embedding"), 0);
    }
    auto inputs = make_genai_inputs(3);
    const std::map<std::string, ov::Tensor> lookup_inputs{{"input_ids", inputs.at("input_ids")}};
    for (const auto& model : {m.model, restored}) {
        AdaptToGenAI adapter(AdaptToGenAI::InputMode::EMBEDS_TO_LOGITS);
        ASSERT_TRUE(adapter.run_on_model(model));
        ASSERT_EQ(find_parameter(model, "input_ids"), nullptr);
        const auto embedding_model = adapter.get_embedding_model();
        ASSERT_NE(embedding_model, nullptr);
        const auto embeddings = run_on_cpu(embedding_model, lookup_inputs);
        ASSERT_EQ(embeddings.get_shape(), (ov::Shape{1, 3, 4}));
        for (size_t i = 0; i < embeddings.get_size(); ++i)
            EXPECT_FLOAT_EQ(embeddings.data<const float>()[i], 2.f * static_cast<float>(i));
        if (per_layer) {
            EXPECT_EQ(model->input("per_layer_inputs").get_partial_shape(), (ov::PartialShape{-1, -1, 3, 2}));
            const auto per_layer_model = adapter.get_per_layer_embedding_model();
            ASSERT_NE(per_layer_model, nullptr);
            const auto per_layer_embeddings = run_on_cpu(per_layer_model, lookup_inputs);
            ASSERT_EQ(per_layer_embeddings.get_shape(), (ov::Shape{1, 3, 3, 2}));
            for (size_t i = 0; i < per_layer_embeddings.get_size(); ++i)
                EXPECT_NEAR(per_layer_embeddings.data<const float>()[i],
                            std::sqrt(2.f) * (100.f + static_cast<float>(i)),
                            1e-4f);
        }
        auto decoder_inputs = inputs;
        decoder_inputs.erase("input_ids");
        ov::Tensor injected(ov::element::f32, {1, 3, 4});
        std::fill_n(injected.data<float>(), injected.get_size(), 7.f);
        decoder_inputs.emplace("inputs_embeds", injected);
        ov::Tensor token_types(ov::element::i64, {1, 3});
        std::fill_n(token_types.data<int64_t>(), token_types.get_size(), 1);
        decoder_inputs.emplace("token_type_ids", token_types);
        if (per_layer) {
            ov::Tensor injected_per_layer(ov::element::f32, {1, 3, 3, 2});
            std::fill_n(injected_per_layer.data<float>(), injected_per_layer.get_size(), 11.f);
            decoder_inputs.emplace("per_layer_inputs", injected_per_layer);
        }
        const auto actual = run_on_cpu(model, decoder_inputs);
        ASSERT_EQ(actual.get_shape(), (ov::Shape{1, 3, 1}));
        for (size_t i = 0; i < actual.get_size(); ++i)
            EXPECT_FLOAT_EQ(actual.data<const float>()[i], 28.f + (per_layer ? 66.f : 0.f));
    }
}

INSTANTIATE_TEST_SUITE_P(NativeBoundary,
                         GGUFAdaptToGenAILegacyScaledEmbedding,
                         testing::Values(std::make_pair(std::string{"gemma3"}, false),
                                         std::make_pair(std::string{"gemma4"}, true)));

TEST(GGUFAdaptToGenAI, LegacyBoundaryFallbackKeepsExplicitMarkersAuthoritative) {
    auto m = build_scaled_legacy_gguf_model("gemma4", true);
    for (const auto& node : m.model->get_ops()) {
        node->get_rt_info().erase("gguf.token_embedding");
        node->get_rt_info().erase("gguf.per_layer_token_embedding");
    }
    m.embd->get_rt_info()["gguf.token_embedding"] = true;
    m.pe_tok->get_rt_info()["gguf.per_layer_token_embedding"] = int64_t{3};
    AdaptToGenAI adapter(AdaptToGenAI::InputMode::EMBEDS_TO_LOGITS);
    ASSERT_TRUE(adapter.run_on_model(m.model));
    const auto embedding_model = adapter.get_embedding_model();
    auto inputs = make_genai_inputs(3);
    const std::map<std::string, ov::Tensor> lookup_inputs{{"input_ids", inputs.at("input_ids")}};
    const auto embeddings = run_on_cpu(embedding_model, lookup_inputs);
    for (size_t i = 0; i < embeddings.get_size(); ++i)
        EXPECT_FLOAT_EQ(embeddings.data<const float>()[i], static_cast<float>(i));
    const auto per_layer_model = adapter.get_per_layer_embedding_model();
    ASSERT_NE(per_layer_model, nullptr);
    const auto per_layer_embeddings = run_on_cpu(per_layer_model, lookup_inputs);
    for (size_t i = 0; i < per_layer_embeddings.get_size(); ++i)
        EXPECT_FLOAT_EQ(per_layer_embeddings.data<const float>()[i], 100.f + static_cast<float>(i));
}

class GGUFAdaptToGenAILegacyHadamardEmbedding
    : public testing::TestWithParam<std::tuple<bool, ov::element::Type, bool>> {};

TEST_P(GGUFAdaptToGenAILegacyHadamardEmbedding, ReloadExtractsCompleteInverseGetRowsEndpoint) {
    const auto& [signed_transform, weight_type, matrix_endpoint] = GetParam();
    auto m = build_minimal_gguf_model(4, 4);
    std::vector<float> latent_values(16);
    std::iota(latent_values.begin(), latent_values.end(), 1.f);
    const auto table = v0::Constant::create(weight_type, {1, 1, 4, 4}, latent_values);
    table->set_friendly_name("Constant_native_token_embd.weight");
    const std::vector<TensorDesc> input_descriptions{{"token_embd.weight", weight_type, {1, 1, 4, 4}},
                                                     {"inp_tokens", ov::element::i32, {1, 1, 1, -1}}};
    const std::vector<TensorDesc> extra_inputs;
    const TensorDesc output_description{"embd", ov::element::f32, {1, 1, -1, 4}};
    std::map<std::string, ov::Any> attributes{{"hadamard_block", int64_t{4}}, {"hadamard_inverse", true}};
    if (signed_transform) {
        ov::Tensor signs(ov::element::f32, {4});
        const std::vector<float> sign_values{-1.f, 1.f, -1.f, 1.f};
        std::copy(sign_values.begin(), sign_values.end(), signs.data<float>());
        attributes.emplace("hadamard_signs", signs);
    }
    const auto decoder = std::make_shared<SingleOpDecoder>("GGML_OP_GET_ROWS",
                                                           input_descriptions,
                                                           extra_inputs,
                                                           output_description,
                                                           attributes);
    auto tensor_map = std::make_shared<ov::frontend::gguf::TensorMap>();
    (*tensor_map)["token_embd.weight"] = table;
    (*tensor_map)["inp_tokens"] = m.inp_tokens;
    const ov::frontend::gguf::NodeContext context(decoder, tensor_map);
    const auto translated = ov::frontend::gguf::op::translate_get_rows(context);
    ASSERT_EQ(translated.size(), 1);
    auto endpoint = translated.front().get_node_shared_ptr();
    if (matrix_endpoint) {
        endpoint->set_friendly_name("inverse_embedding");
        const auto identity =
            v0::Constant::create(ov::element::f32, {4, 4}, {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1});
        const auto matrix = std::make_shared<v0::MatMul>(endpoint, identity, false, false);
        matrix->set_friendly_name("embd");
        endpoint = matrix;
    }
    endpoint->get_rt_info()["gguf.token_embedding"] = true;
    const auto logits = m.model->get_results().front()->get_input_node_shared_ptr(0);
    logits->input(0).replace_source_output(endpoint);
    m.model->get_rt_info()["gguf_architecture"] = std::string{"qwen35"};
    m.model->validate_nodes_and_infer_types();
    const auto restored = reload_ir_model(m.model);
    for (const auto& node : restored->get_ops())
        ASSERT_EQ(node->get_rt_info().count("gguf.token_embedding"), 0);
    AdaptToGenAI direct_adapter(AdaptToGenAI::InputMode::EMBEDS_TO_LOGITS);
    AdaptToGenAI restored_adapter(AdaptToGenAI::InputMode::EMBEDS_TO_LOGITS);
    ASSERT_TRUE(direct_adapter.run_on_model(m.model));
    ASSERT_TRUE(restored_adapter.run_on_model(restored));
    auto inputs = make_genai_inputs(3);
    const std::map<std::string, ov::Tensor> lookup_inputs{{"input_ids", inputs.at("input_ids")}};
    const auto direct_embedding_model = direct_adapter.get_embedding_model();
    const auto restored_embedding_model = restored_adapter.get_embedding_model();
    ASSERT_NE(direct_embedding_model, nullptr);
    ASSERT_NE(restored_embedding_model, nullptr);
    const auto direct_embeddings = run_on_cpu(direct_embedding_model, lookup_inputs);
    const auto restored_embeddings = run_on_cpu(restored_embedding_model, lookup_inputs);
    ASSERT_EQ(direct_embeddings.get_shape(), (ov::Shape{1, 3, 4}));
    ASSERT_EQ(restored_embeddings.get_shape(), direct_embeddings.get_shape());
    float latent_difference = 0.f;
    for (size_t i = 0; i < direct_embeddings.get_size(); ++i) {
        EXPECT_NEAR(restored_embeddings.data<const float>()[i], direct_embeddings.data<const float>()[i], 1e-4f);
        latent_difference += std::abs(direct_embeddings.data<const float>()[i] - latent_values[i]);
    }
    EXPECT_GT(latent_difference, 1.f);
    inputs.erase("input_ids");
    ov::Tensor injected(ov::element::f32, {1, 3, 4});
    const std::vector<float> injected_values{7.f, 11.f, 13.f, 17.f};
    for (size_t i = 0; i < injected.get_size(); ++i)
        injected.data<float>()[i] = injected_values[i % 4];
    inputs.emplace("inputs_embeds", injected);
    for (const auto& model : {m.model, restored}) {
        EXPECT_EQ(find_parameter(model, "input_ids"), nullptr);
        const auto actual = run_on_cpu(model, inputs);
        ASSERT_EQ(actual.get_shape(), (ov::Shape{1, 3, 1}));
        for (size_t i = 0; i < actual.get_size(); ++i)
            EXPECT_FLOAT_EQ(actual.data<const float>()[i], 48.f);
    }
}

INSTANTIATE_TEST_SUITE_P(InverseSignsAndWeightType,
                         GGUFAdaptToGenAILegacyHadamardEmbedding,
                         testing::Combine(testing::Bool(),
                                          testing::Values(ov::element::f32, ov::element::f16),
                                          testing::Bool()));

TEST(GGUFAdaptToGenAI, LegacyCompleteEmbeddingEndpointRejectsSeparateBranches) {
    auto m = build_minimal_gguf_model();
    const auto lookup = m.embd->get_input_node_shared_ptr(0);
    const auto table = lookup->get_input_node_shared_ptr(0);
    table->set_friendly_name("token_embd.weight");
    const auto left = std::make_shared<v0::Convert>(m.embd, ov::element::f32);
    const auto right = std::make_shared<v0::Convert>(m.embd, ov::element::f32);
    left->set_friendly_name("Convert_left_embd");
    right->set_friendly_name("Convert_right_embd");
    const auto left_result = std::make_shared<v0::Result>(left);
    const auto right_result = std::make_shared<v0::Result>(right);
    m.model->add_results({left_result, right_result});
    const auto restored = reload_ir_model(m.model);
    AdaptToGenAI adapter(AdaptToGenAI::InputMode::EMBEDS_TO_LOGITS);
    EXPECT_THROW(adapter.run_on_model(restored), ov::Exception);
}

TEST(GGUFAdaptToGenAI, LegacyCompleteEmbeddingEndpointIgnoresPerLayerSuffix) {
    auto m = build_minimal_gguf_model();
    const auto lookup = m.embd->get_input_node_shared_ptr(0);
    const auto table = lookup->get_input_node_shared_ptr(0);
    table->set_friendly_name("token_embd.weight");
    const auto per_layer = std::make_shared<v0::Convert>(m.embd, ov::element::f32);
    per_layer->set_friendly_name("Convert_native_per_layer_embd");
    const auto result = std::make_shared<v0::Result>(per_layer);
    m.model->add_results({result});
    const auto restored = reload_ir_model(m.model);
    const auto legacy = ov::frontend::gguf::resolve_legacy_embedding_boundaries(restored, true, false);
    ASSERT_NE(legacy.token, nullptr);
    EXPECT_EQ(legacy.token->get_friendly_name(), m.embd->get_friendly_name());
}

TEST(GGUFAdaptToGenAI, RecurrentBatchingUsesDeclaredContract) {
    for (const bool batchable : {false, true}) {
        auto model = build_minimal_gguf_model().model;
        model->get_rt_info()[ov::frontend::gguf::pass::gguf_recurrent_states_key()] = std::vector<std::string>{};
        model->get_rt_info()["gguf_architecture"] = std::string("custom-recurrent-decoder");
        ov::frontend::gguf::ModelContract contract;
        contract.batchable_recurrent = batchable;
        ov::frontend::gguf::write_model_contract(model, contract);
        ASSERT_TRUE(AdaptToGenAI().run_on_model(model));
        EXPECT_EQ(model->input("input_ids").get_partial_shape()[0],
                  batchable ? ov::Dimension::dynamic() : ov::Dimension{1});
    }
}

TEST(GGUFAdaptToGenAI, ExplicitRecurrentContractOverridesGatedDeltaNetInference) {
    for (const auto batchable : std::vector<std::optional<bool>>{std::nullopt, false, true}) {
        auto model = build_minimal_gguf_model().model;
        const auto data = v0::Constant::create(ov::element::f32, {1, 1, 1, 1}, {1.f});
        const auto gate = v0::Constant::create(ov::element::f32, {1, 1, 1}, {1.f});
        const auto gdn = std::make_shared<ov::op::internal::GatedDeltaNet>(data, data, data, data, gate, gate);
        const auto axes = v0::Constant::create(ov::element::i64, {4}, {0, 1, 2, 3});
        const auto sum = std::make_shared<v1::ReduceSum>(gdn->output(0), axes, false);
        const auto result = model->get_results().front();
        const auto combined = std::make_shared<v1::Add>(result->input_value(0), sum);
        result->input(0).replace_source_output(combined);
        model->get_rt_info()[ov::frontend::gguf::pass::gguf_recurrent_states_key()] = std::vector<std::string>{};
        ov::frontend::gguf::ModelContract contract;
        contract.batchable_recurrent = batchable;
        ov::frontend::gguf::write_model_contract(model, contract);
        EXPECT_EQ(ov::frontend::gguf::read_model_contract(model).batchable_recurrent, batchable);
        ASSERT_TRUE(AdaptToGenAI().run_on_model(model));
        EXPECT_EQ(model->input("input_ids").get_partial_shape()[0],
                  batchable.value_or(true) ? ov::Dimension::dynamic() : ov::Dimension{1});
    }
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
    ov::InferRequest per_layer_lookup;
    if (per_layer)
        per_layer_lookup = core.compile_model(adapter.get_per_layer_embedding_model(), "CPU").create_infer_request();
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
            if (per_layer) {
                per_layer_lookup.set_tensor("input_ids", inputs.at("input_ids"));
                per_layer_lookup.infer();
                values.set_tensor("per_layer_inputs", per_layer_lookup.get_tensor("per_layer_inputs"));
            }
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
