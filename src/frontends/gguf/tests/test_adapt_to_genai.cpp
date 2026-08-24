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

#include "gtest/gtest.h"
#include "op_test_utils.hpp"
#include "openvino/frontend/gguf/adapt_to_genai.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"

using namespace ov_gguf_test;
using ov::frontend::gguf::pass::AdaptToGenAI;

namespace {

std::shared_ptr<ov::op::v0::Constant> const_i64(const std::vector<int64_t>& values) {
    return ov::op::v0::Constant::create(ov::element::i64, ov::Shape{values.size()}, values);
}

std::shared_ptr<ov::op::v0::Parameter> find_param(const std::shared_ptr<ov::Model>& model, const std::string& name) {
    for (const auto& p : model->get_parameters()) {
        if (p->output(0).get_names().count(name) || p->get_friendly_name() == name) {
            return p;
        }
    }
    return nullptr;
}

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
    std::shared_ptr<ov::op::v0::Parameter> inp_tokens;
    std::shared_ptr<ov::Node> embd;  // Unsqueeze(Gather(vocab, Squeeze(inp_tokens)), axis=0)
    std::shared_ptr<ov::op::v0::Parameter> inp_out_ids;
    std::shared_ptr<ov::Node> row_select;  // Unsqueeze(Gather(Squeeze(embd), Squeeze(inp_out_ids)), axis=0)
};

MinimalGgufModel build_minimal_gguf_model(int64_t vocab = 4, int64_t hidden = 2, bool with_inp_out_ids = false) {
    MinimalGgufModel m;

    m.inp_tokens = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{1, 1, 1, -1});
    m.inp_tokens->output(0).set_names({"inp_tokens"});
    auto inp_pos = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{1, 1, 1, -1});
    inp_pos->output(0).set_names({"inp_pos"});
    auto self_kq_mask = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 1, -1, -1});
    self_kq_mask->output(0).set_names({"self_kq_mask"});
    auto token_len_per_seq = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{1});
    token_len_per_seq->output(0).set_names({"token_len_per_seq"});
    auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1});
    beam_idx->output(0).set_names({"beam_idx"});

    std::vector<float> table_values(vocab * hidden);
    for (size_t i = 0; i < table_values.size(); ++i) {
        table_values[i] = static_cast<float>(i);
    }
    auto vocab_table = ov::op::v0::Constant::create(ov::element::f32, {(size_t)vocab, (size_t)hidden}, table_values);

    // Mirrors translate_get_rows's embedding-table ("else", rank-2 data) branch exactly:
    // Squeeze(indices, [0,1]) -> Gather(table, ., axis=0) -> Unsqueeze(., axis=0).
    auto indices = std::make_shared<ov::op::v0::Squeeze>(m.inp_tokens, const_i64({0, 1}));
    auto gather = std::make_shared<ov::op::v8::Gather>(vocab_table, indices, const_i64({0}));
    m.embd = std::make_shared<ov::op::v0::Unsqueeze>(gather, const_i64({0}));
    m.embd->set_friendly_name("Unsqueeze_test_embd");

    ov::ParameterVector params{m.inp_tokens, inp_pos, self_kq_mask, token_len_per_seq, beam_idx};
    if (with_inp_out_ids) {
        m.inp_out_ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{1, 1, 1, -1});
        m.inp_out_ids->output(0).set_names({"inp_out_ids"});
        params.push_back(m.inp_out_ids);

        // Mirrors translate_get_rows's rank-4/dim1==1 branch ("attn_out_g"/"inpSA_g" in a real
        // model): Squeeze(data,[0,1]) -> Gather(., Squeeze(indices,[0,1]), axis=0) -> Unsqueeze(0).
        auto data_squeeze = std::make_shared<ov::op::v0::Squeeze>(m.embd, const_i64({0, 1}));
        auto row_indices = std::make_shared<ov::op::v0::Squeeze>(m.inp_out_ids, const_i64({0, 1}));
        auto row_gather = std::make_shared<ov::op::v8::Gather>(data_squeeze, row_indices, const_i64({0}));
        m.row_select = std::make_shared<ov::op::v0::Unsqueeze>(row_gather, const_i64({0}));
        m.row_select->set_friendly_name("Unsqueeze_test_row_select");
    }

    // A trivial rank-4 "logits" output so AdaptToGenAI's final logits reshape has something to
    // work on: reduce the hidden axis so the shape stays predictable regardless of vocab/hidden.
    auto logits_src = with_inp_out_ids ? m.row_select : m.embd;
    auto logits = std::make_shared<ov::op::v1::ReduceSum>(logits_src, const_i64({3}), true);
    auto result = std::make_shared<ov::op::v0::Result>(logits);

    m.model = std::make_shared<ov::Model>(ov::ResultVector{result}, params);
    return m;
}

}  // namespace

// AdaptToGenAI rewrites the gguf-style IO into genai's input_ids/attention_mask/position_ids
// contract, and beam_idx passes through unchanged (genai sets it directly).
TEST(GGUFAdaptToGenAI, RewritesIOContract) {
    auto m = build_minimal_gguf_model();

    ASSERT_TRUE(AdaptToGenAI().run_on_model(m.model));

    auto input_ids = find_param(m.model, "input_ids");
    auto attention_mask = find_param(m.model, "attention_mask");
    auto position_ids = find_param(m.model, "position_ids");
    auto beam_idx = find_param(m.model, "beam_idx");
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

// Without the required gguf inputs present, the pass is a no-op (e.g. a model already adapted, or
// not a gguf-IO model at all).
TEST(GGUFAdaptToGenAI, NoOpWithoutGgufInputs) {
    auto input_ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{-1, -1});
    auto result = std::make_shared<ov::op::v0::Result>(input_ids);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input_ids});

    EXPECT_FALSE(AdaptToGenAI().run_on_model(model));
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

    auto axis_const = ov::as_type_ptr<ov::op::v0::Constant>(fixed_embd->input_value(1).get_node_shared_ptr());
    ASSERT_NE(axis_const, nullptr);
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
    auto embd_result = std::make_shared<ov::op::v0::Result>(m.embd);
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

    auto gather = ov::as_type_ptr<ov::op::v8::Gather>(fixed_row_select->input_value(0).get_node_shared_ptr());
    ASSERT_NE(gather, nullptr);

    // Both the activation and the indices are now fed through a Reshape (flatten), not a Squeeze.
    auto data_flat = ov::as_type_ptr<ov::op::v1::Reshape>(gather->input_value(0).get_node_shared_ptr());
    auto indices_flat = ov::as_type_ptr<ov::op::v1::Reshape>(gather->input_value(1).get_node_shared_ptr());
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
    auto row_result = std::make_shared<ov::op::v0::Result>(m.row_select);
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
