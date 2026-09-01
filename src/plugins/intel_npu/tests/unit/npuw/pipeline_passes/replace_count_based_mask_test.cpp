// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "npuw_transformations/replace_count_based_mask.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/cum_sum.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/logical_or.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/runtime/tensor.hpp"

namespace {

using namespace ov;

constexpr int64_t kHeads = 2;

template <typename T>
size_t count_ops(const std::shared_ptr<ov::Model>& model) {
    size_t n = 0;
    for (const auto& op : model->get_ops()) {
        if (ov::is_type<T>(op)) {
            ++n;
        }
    }
    return n;
}

// Rebuilds the count-based attention mask that onnxruntime-genai emits for
// GroupQueryAttention, as observed in the exported Qwen3-4B IR:
//
//   attention_mask -> Convert -> ReduceSum ---> Subtract(S) --> past'
//                                                                 |
//   ones[S] -> CumSum(inclusive) -> rows (i+1) ------------> Add -> bound[i] = past' + i + 1
//                                                                 |
//   ones[1,H,S,T] -> CumSum(exclusive, axis=-1) -> j ---> Less ----+
//                                                          |
//                              Select(less, 0, -inf) -> Add(scores) -> Result
//
// The real export spreads bound over Gather/ReduceMax(Range)/Broadcast/Transpose; the
// arithmetic reproduced here is the same and the pass does not match on those nodes.
// `layers` chains repeat the per-layer part over one shared attention_mask, as in the
// 36-layer model.
//
// The index chain defaults to f32 because CumSum only has an f32 reference evaluator and
// the semantic tests below need to run the unfixed graph too; the export emits i32, which
// MatchesTheExportedInt32Chain covers.
std::shared_ptr<ov::Model> make_count_mask_model(int64_t S,
                                                 int64_t T,
                                                 size_t layers = 1,
                                                 bool bound_from_mask = true,
                                                 int64_t mask_len = -1,
                                                 const std::string& mask_name = "attention_mask",
                                                 const ov::element::Type& index_et = ov::element::f32) {
    if (mask_len < 0) {
        mask_len = T;
    }
    ov::ParameterVector params;

    auto mask = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{1, mask_len});
    mask->set_friendly_name(mask_name);
    mask->output(0).set_names({mask_name});
    params.push_back(mask);

    ov::Output<ov::Node> past;  // past' = sum(attention_mask) - S
    if (bound_from_mask) {
        auto cvt = std::make_shared<ov::op::v0::Convert>(mask, index_et);
        auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
        auto total = std::make_shared<ov::op::v1::ReduceSum>(cvt, axes, false);
        auto s_const = ov::op::v0::Constant::create(index_et, ov::Shape{1}, {S});
        past = std::make_shared<ov::op::v1::Subtract>(total, s_const)->output(0);
    } else {
        // Same shape of graph, but the bound is a plain runtime input: not the broken idiom.
        auto past_param = std::make_shared<ov::op::v0::Parameter>(index_et, ov::PartialShape{1});
        past_param->set_friendly_name("past_len");
        params.push_back(past_param);
        past = past_param->output(0);
    }

    ov::ResultVector results;
    for (size_t l = 0; l < layers; ++l) {
        const std::string tag = std::to_string(l);

        auto ones_s = ov::op::v0::Constant::create(index_et,
                                                   ov::Shape{static_cast<size_t>(S)},
                                                   std::vector<int32_t>(static_cast<size_t>(S), 1));
        auto axis_0 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
        auto rows = std::make_shared<ov::op::v0::CumSum>(ones_s, axis_0, /*exclusive=*/false, /*reverse=*/false);
        auto bound_1d = std::make_shared<ov::op::v1::Add>(rows, past);
        auto bound_shape =
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{1, 1, S, 1});
        auto bound = std::make_shared<ov::op::v1::Reshape>(bound_1d, bound_shape, false);
        bound->set_friendly_name("bound." + tag);

        auto one = ov::op::v0::Constant::create(index_et, ov::Shape{1}, {1});
        auto j_shape =
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{1, kHeads, S, T});
        auto ones_4d = std::make_shared<ov::op::v3::Broadcast>(one, j_shape);
        auto axis_3 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {3});
        auto key_index = std::make_shared<ov::op::v0::CumSum>(ones_4d, axis_3, /*exclusive=*/true, /*reverse=*/false);

        auto less = std::make_shared<ov::op::v1::Less>(key_index, bound);
        auto visible = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {0.0f});
        auto masked =
            ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {-std::numeric_limits<float>::infinity()});
        auto select = std::make_shared<ov::op::v1::Select>(less, visible, masked);
        select->set_friendly_name("attn_mask_select." + tag);

        auto scores = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, kHeads, S, T});
        scores->set_friendly_name("scores." + tag);
        scores->output(0).set_names({"scores." + tag});
        params.push_back(scores);

        auto add = std::make_shared<ov::op::v1::Add>(scores, select);
        auto result = std::make_shared<ov::op::v0::Result>(add);
        result->set_friendly_name("masked_scores." + tag);
        results.push_back(result);
    }

    return std::make_shared<ov::Model>(results, params, "count_mask_model");
}

// Runs the mask through the reference evaluator and reports, per query row, which keys it
// can actually attend to. Scores are fed as zeros, so a visible key reads back as 0 and a
// masked one as -inf. This is the CI form of the `--mask-audit` check from the report.
std::vector<std::set<int64_t>> visible_keys(const std::shared_ptr<ov::Model>& model,
                                            const std::vector<int64_t>& mask,
                                            size_t layer = 0) {
    ov::TensorVector inputs;
    for (const auto& param : model->get_parameters()) {
        ov::Tensor t(param->get_element_type(), param->get_shape());
        if (param->get_element_type() == ov::element::i64 && t.get_size() == mask.size()) {
            std::copy(mask.begin(), mask.end(), t.data<int64_t>());
        } else {
            std::memset(t.data(), 0, t.get_byte_size());
        }
        inputs.push_back(t);
    }

    ov::TensorVector outputs;
    for (const auto& result : model->get_results()) {
        outputs.emplace_back(result->get_output_element_type(0), result->get_output_shape(0));
    }
    EXPECT_TRUE(model->evaluate(outputs, inputs));

    const auto& out = outputs.at(layer);
    const auto& shape = out.get_shape();  // [1, H, S, T]
    const int64_t S = static_cast<int64_t>(shape[2]);
    const int64_t T = static_cast<int64_t>(shape[3]);
    const float* data = out.data<float>();

    std::vector<std::set<int64_t>> per_row(static_cast<size_t>(S));
    for (int64_t i = 0; i < S; ++i) {
        for (int64_t j = 0; j < T; ++j) {
            if (std::isfinite(data[i * T + j])) {  // head 0
                per_row[static_cast<size_t>(i)].insert(j);
            }
        }
    }
    return per_row;
}

// The contract the rewritten mask implements: query row i owns key position (T-S)+i and
// attends to every valid key at or before it. Rows that carry no real token (prompt
// padding, unused generation slots) keep their own slot so the softmax row is not all -inf.
std::vector<std::set<int64_t>> expected_visible_keys(const std::vector<int64_t>& mask, int64_t S, int64_t T) {
    const int64_t off = T - S;
    std::vector<std::set<int64_t>> expected(static_cast<size_t>(S));
    for (int64_t i = 0; i < S; ++i) {
        const int64_t i_abs = off + i;
        for (int64_t j = 0; j <= i_abs; ++j) {
            if (mask[static_cast<size_t>(j)] != 0) {
                expected[static_cast<size_t>(i)].insert(j);
            }
        }
        expected[static_cast<size_t>(i)].insert(i_abs);
    }
    return expected;
}

// ---------------------------------------------------------------------------------------
// Prefill: a 3-token prompt left-padded into an 8-slot window (NPUW right-aligns the real
// tokens). sum(attention_mask) == 3 != past_len + S == 8, so the count-based bound is off
// by the pad count.
// ---------------------------------------------------------------------------------------

TEST(ReplaceCountBasedMask, PrefillLeftPaddedPromptIsBrokenBeforeTheFix) {
    const int64_t S = 8, T = 8;
    const std::vector<int64_t> mask{0, 0, 0, 0, 0, 1, 1, 1};

    auto model = make_count_mask_model(S, T);
    const auto actual = visible_keys(model, mask);

    // bound[i] = (3 - 8) + i + 1 = i - 4, so the first real token (row 5) sees key 0 only -
    // a pad slot - and no real token can attend to itself.
    EXPECT_EQ(actual[5], (std::set<int64_t>{0}));
    EXPECT_EQ(actual[6], (std::set<int64_t>{0, 1}));
    EXPECT_EQ(actual[7], (std::set<int64_t>{0, 1, 2}));
}

TEST(ReplaceCountBasedMask, PrefillLeftPaddedPrompt) {
    const int64_t S = 8, T = 8;
    const std::vector<int64_t> mask{0, 0, 0, 0, 0, 1, 1, 1};

    auto model = make_count_mask_model(S, T);
    ASSERT_TRUE(ov::npuw::ReplaceCountBasedMask().run_on_model(model));

    const auto actual = visible_keys(model, mask);
    EXPECT_EQ(actual, expected_visible_keys(mask, S, T));

    // Spelled out: real tokens attend causally to the real prefix, pad rows keep one key.
    EXPECT_EQ(actual[5], (std::set<int64_t>{5}));
    EXPECT_EQ(actual[6], (std::set<int64_t>{5, 6}));
    EXPECT_EQ(actual[7], (std::set<int64_t>{5, 6, 7}));
    for (int64_t i = 0; i < 5; ++i) {
        EXPECT_EQ(actual[static_cast<size_t>(i)], (std::set<int64_t>{i})) << "pad row " << i;
    }
}

TEST(ReplaceCountBasedMask, PrefillFullWindowIsUnaffected) {
    // The geometry CPU/GPU run: no slack, so the count-based mask was already correct.
    // The rewrite must reproduce it exactly.
    const int64_t S = 8, T = 8;
    const std::vector<int64_t> mask(8, 1);

    auto model = make_count_mask_model(S, T);
    const auto before = visible_keys(model, mask);
    ASSERT_TRUE(ov::npuw::ReplaceCountBasedMask().run_on_model(model));
    const auto after = visible_keys(model, mask);

    EXPECT_EQ(before, after);
    EXPECT_EQ(after, expected_visible_keys(mask, S, T));
}

TEST(ReplaceCountBasedMask, ChunkedPrefillChunkIsRightAlignedInTheKeyWindow) {
    // 2 tokens already prefilled at key slots [0, 2), a 3-token chunk right-aligned in a
    // 4-slot query window (its first row is padding).
    const int64_t S = 4, T = 8;
    const std::vector<int64_t> mask{1, 1, 0, 0, 0, 1, 1, 1};

    auto model = make_count_mask_model(S, T);
    ASSERT_TRUE(ov::npuw::ReplaceCountBasedMask().run_on_model(model));

    const auto actual = visible_keys(model, mask);
    EXPECT_EQ(actual, expected_visible_keys(mask, S, T));
    EXPECT_EQ(actual[0], (std::set<int64_t>{0, 1, 4}));  // pad row: stored past + its own slot
    EXPECT_EQ(actual[1], (std::set<int64_t>{0, 1, 5}));  // first real token of the chunk
    EXPECT_EQ(actual[2], (std::set<int64_t>{0, 1, 5, 6}));
    EXPECT_EQ(actual[3], (std::set<int64_t>{0, 1, 5, 6, 7}));
}

// ---------------------------------------------------------------------------------------
// Generate: the past KV window is fixed while only num_stored_tokens slots are valid, and
// the new token is appended at the END of the key axis.
// ---------------------------------------------------------------------------------------

TEST(ReplaceCountBasedMask, GenerateCannotAttendToItselfBeforeTheFix) {
    const int64_t S = 1, T = 8;
    const std::vector<int64_t> mask{1, 1, 1, 0, 0, 0, 0, 1};  // 3 stored + current at slot 7

    auto model = make_count_mask_model(S, T);
    const auto actual = visible_keys(model, mask);

    // Right count (4 keys), wrong set: stale slot 3 is in, the current token at 7 is out -
    // the token cannot see itself, so the model re-emits it.
    EXPECT_EQ(actual[0], (std::set<int64_t>{0, 1, 2, 3}));
}

TEST(ReplaceCountBasedMask, GenerateSingleToken) {
    const int64_t S = 1, T = 8;
    const std::vector<int64_t> mask{1, 1, 1, 0, 0, 0, 0, 1};

    auto model = make_count_mask_model(S, T);
    ASSERT_TRUE(ov::npuw::ReplaceCountBasedMask().run_on_model(model));

    const auto actual = visible_keys(model, mask);
    EXPECT_EQ(actual, expected_visible_keys(mask, S, T));
    EXPECT_EQ(actual[0], (std::set<int64_t>{0, 1, 2, 7}));
}

TEST(ReplaceCountBasedMask, GenerateMultipleTokens) {
    // max_generation_token_len == 2: both new tokens sit at the tail of the key axis.
    const int64_t S = 2, T = 8;
    const std::vector<int64_t> mask{1, 1, 1, 0, 0, 0, 1, 1};

    auto model = make_count_mask_model(S, T);
    ASSERT_TRUE(ov::npuw::ReplaceCountBasedMask().run_on_model(model));

    const auto actual = visible_keys(model, mask);
    EXPECT_EQ(actual, expected_visible_keys(mask, S, T));
    EXPECT_EQ(actual[0], (std::set<int64_t>{0, 1, 2, 6}));
    EXPECT_EQ(actual[1], (std::set<int64_t>{0, 1, 2, 6, 7}));
}

// ---------------------------------------------------------------------------------------
// Graph-level effects
// ---------------------------------------------------------------------------------------

TEST(ReplaceCountBasedMask, ConditionIsSharedAndTheBoundChainIsDropped) {
    const int64_t S = 8, T = 8;
    auto model = make_count_mask_model(S, T, /*layers=*/3);

    ASSERT_EQ(count_ops<ov::op::v1::Select>(model), 3u);
    ASSERT_EQ(count_ops<ov::op::v0::CumSum>(model), 6u);

    ASSERT_TRUE(ov::npuw::ReplaceCountBasedMask().run_on_model(model));

    // One condition for all layers, so partitioning sees a single common subexpression.
    EXPECT_EQ(count_ops<ov::op::v1::LogicalAnd>(model), 1u);
    EXPECT_EQ(count_ops<ov::op::v1::LogicalOr>(model), 1u);
    EXPECT_EQ(count_ops<ov::op::v1::Select>(model), 3u);
    for (const auto& result : model->get_results()) {
        const auto& select = result->input_value(0).get_node()->input_value(1);
        EXPECT_TRUE(ov::is_type<ov::op::v1::Select>(select.get_node()));
        EXPECT_TRUE(ov::is_type<ov::op::v1::LogicalOr>(select.get_node()->input_value(0).get_node()));
    }

    // The count-based bound - including the [1, H, S, T] i32 CumSum per layer - is dead.
    EXPECT_EQ(count_ops<ov::op::v0::CumSum>(model), 0u);
    EXPECT_EQ(count_ops<ov::op::v1::Less>(model), 0u);
    EXPECT_EQ(count_ops<ov::op::v1::ReduceSum>(model), 0u);

    // The mask itself is no longer materialized per head.
    for (const auto& result : model->get_results()) {
        const auto& select = result->input_value(0).get_node()->input_value(1);
        EXPECT_EQ(select.get_shape(), (ov::Shape{1, 1, 8, 8}));
    }
}

TEST(ReplaceCountBasedMask, MatchesTheExportedInt32Chain) {
    // The exported graph builds the key index and the bound in i32. Match-only: CumSum has
    // no i32 reference evaluator, so the unfixed graph cannot be run here.
    auto model = make_count_mask_model(8,
                                       8,
                                       /*layers=*/2,
                                       /*bound_from_mask=*/true,
                                       /*mask_len=*/-1,
                                       /*mask_name=*/"attention_mask",
                                       /*index_et=*/ov::element::i32);

    ASSERT_TRUE(ov::npuw::ReplaceCountBasedMask().run_on_model(model));
    EXPECT_EQ(count_ops<ov::op::v1::LogicalAnd>(model), 1u);
    EXPECT_EQ(count_ops<ov::op::v0::CumSum>(model), 0u);
}

TEST(ReplaceCountBasedMask, AllLayersAgreeOnTheSameMask) {
    const int64_t S = 1, T = 8;
    const std::vector<int64_t> mask{1, 1, 1, 0, 0, 0, 0, 1};

    auto model = make_count_mask_model(S, T, /*layers=*/3);
    ASSERT_TRUE(ov::npuw::ReplaceCountBasedMask().run_on_model(model));

    const auto expected = expected_visible_keys(mask, S, T);
    for (size_t layer = 0; layer < 3; ++layer) {
        EXPECT_EQ(visible_keys(model, mask, layer), expected) << "layer " << layer;
    }
}

// ---------------------------------------------------------------------------------------
// Guards: anything that is not the count-based idiom must be left alone.
// ---------------------------------------------------------------------------------------

TEST(ReplaceCountBasedMask, NoOpWithoutAttentionMask) {
    auto model = make_count_mask_model(8, 8, 1, /*bound_from_mask=*/true, /*mask_len=*/-1, /*mask_name=*/"input_mask");

    EXPECT_FALSE(ov::npuw::ReplaceCountBasedMask().run_on_model(model));
    EXPECT_EQ(count_ops<ov::op::v1::Less>(model), 1u);
}

TEST(ReplaceCountBasedMask, NoOpWhenTheBoundIsNotDerivedFromTheMask) {
    // Same Select(cond, 0, -inf) over a key-index CumSum, but the bound comes from a plain
    // input instead of ReduceSum(attention_mask) - rewriting it would change semantics.
    auto model = make_count_mask_model(8, 8, 1, /*bound_from_mask=*/false);

    EXPECT_FALSE(ov::npuw::ReplaceCountBasedMask().run_on_model(model));
    EXPECT_EQ(count_ops<ov::op::v1::Less>(model), 1u);
    EXPECT_EQ(count_ops<ov::op::v1::LogicalAnd>(model), 0u);
}

TEST(ReplaceCountBasedMask, NoOpWhenTheMaskDoesNotCoverTheKeyWindow) {
    // attention_mask shorter than the key window: it cannot be indexed by key position.
    auto model = make_count_mask_model(8, 8, 1, /*bound_from_mask=*/true, /*mask_len=*/6);

    EXPECT_FALSE(ov::npuw::ReplaceCountBasedMask().run_on_model(model));
    EXPECT_EQ(count_ops<ov::op::v1::Less>(model), 1u);
}

}  // namespace
