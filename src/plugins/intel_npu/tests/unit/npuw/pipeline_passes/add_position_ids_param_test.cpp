// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "npuw_transformations/add_position_ids_param.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <queue>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "openvino/op/add.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/less_eq.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"

namespace {
// Shared builder for LFM2-like RoPE / causal-mask patterns.
//
// Before transformers==5.0.0, the pattern looked like this:
//   Range -> Unsqueeze(0) -> Unsqueeze(1) -> Convert -> MatMul(inv_freq, .)
//                                                     -> Transpose -> Concat(.,.) -> {Cos, Sin}
//                                        `-> Unsqueeze(2) -> LessEqual  (causal mask)
//
// And, there was an additional Range consumer for Gated Short Convolution indexing attached:
//   Range -> Clamp -> Add -> Mod -> Unsqueeze -> ScatterNDUpdate
//
// From transformers>=5.0.0, the pattern is simplified to the first part only (no Clamp consumer).
//
//
// From transformers>=5.4, pattern doesn't use the same subgraph for causal mask and
// RoPE anymore as well as it doesn't utilize Gated Short Convolution Indexing path too.
// So, only the RoPE chain is expected to be matched to feed `position_ids` into.
//
//
// `offset_positions` selects the shape, where Range starts from zero and the past
// length is added on top of it rather than being folded into Range's start.
// `with_range_clamp_consumer` selects whether to attach the additional Range consumer,
// which full pattern is simplified to Range -> Clamp -> Result. It defaults to true, so that a
// no-argument call still builds the pre-5.0.0 IR the older tests below are written against.
std::shared_ptr<ov::Model> build_model_with_lfm2_like_pattern(bool offset_positions = false,
                                                              bool with_range_clamp_consumer = true,
                                                              const std::string& model_name = "lfm2_like_model") {
    // Range: start=0, stop=seq_len, step=1  (mimics position_ids generation)
    auto start = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
    auto stop = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {128});
    auto step = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
    auto range = std::make_shared<ov::op::v4::Range>(start, stop, step, ov::element::i64);
    range->set_friendly_name("range");

    // transformers>=5.4: past length added after the Range instead of folded into its start
    ov::Output<ov::Node> positions = range->output(0);
    if (offset_positions) {
        auto past_len = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {8});
        auto offset_add = std::make_shared<ov::op::v1::Add>(range, past_len);
        offset_add->set_friendly_name("positions_offset_add");
        positions = offset_add->output(0);
    }

    // Unsqueeze: add batch dim [seq_len] → [1, seq_len]
    auto unsqueeze_axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
    auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(positions, unsqueeze_axes);
    unsqueeze->set_friendly_name("unsqueeze_batch");

    // Unsqueeze1: add feature dim [1, seq_len] -> [1, 1, seq_len]
    auto unsqueeze1_axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    auto unsqueeze1 = std::make_shared<ov::op::v0::Unsqueeze>(unsqueeze, unsqueeze1_axes);
    unsqueeze1->set_friendly_name("unsqueeze_feature");

    // Convert (always present in real LFM2 models)
    auto convert = std::make_shared<ov::op::v0::Convert>(unsqueeze1, ov::element::f32);
    convert->set_friendly_name("convert");

    // MatMul: [inv_freq] x [positions] -> freqs
    auto inv_freq = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 8, 1});
    inv_freq->output(0).set_names({"inv_freq"});
    inv_freq->set_friendly_name("inv_freq");

    auto matmul = std::make_shared<ov::op::v0::MatMul>(inv_freq, convert);
    matmul->set_friendly_name("matmul_rope");

    // Transpose: [1, 8, 128] -> [1, 128, 8]
    auto transpose_order = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {0, 2, 1});
    auto transpose = std::make_shared<ov::op::v1::Transpose>(matmul, transpose_order);
    transpose->set_friendly_name("transpose_rope");

    // Concat(transpose, transpose) -> simulate theta doubling
    auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{transpose, transpose}, 2);
    concat->set_friendly_name("concat_theta");

    // Cos / Sin
    auto cos = std::make_shared<ov::op::v0::Cos>(concat);
    cos->set_friendly_name("cos");
    auto sin = std::make_shared<ov::op::v0::Sin>(concat);
    sin->set_friendly_name("sin");

    auto cos_result = std::make_shared<ov::op::v0::Result>(cos);
    cos_result->set_friendly_name("cos_result");
    auto sin_result = std::make_shared<ov::op::v0::Result>(sin);
    sin_result->set_friendly_name("sin_result");

    ov::ResultVector results = {cos_result, sin_result};
    ov::ParameterVector params = {inv_freq};

    // Causal mask consumer:
    // Real LFM2 path: Range -> Unsqueeze -> Unsqueeze -> Unsqueeze -> LessEqual
    auto unsqueeze_causal_axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});
    auto unsqueeze_causal = std::make_shared<ov::op::v0::Unsqueeze>(unsqueeze1, unsqueeze_causal_axes);
    unsqueeze_causal->set_friendly_name("unsqueeze_causal");

    auto stub_k_range_as_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1, 1, 1}, {0});
    auto less_equal = std::make_shared<ov::op::v1::LessEqual>(stub_k_range_as_const, unsqueeze_causal);
    less_equal->set_friendly_name("causal_mask_less_equal");
    auto mask_result = std::make_shared<ov::op::v0::Result>(less_equal);
    mask_result->set_friendly_name("mask_result");
    results.push_back(mask_result);

    if (with_range_clamp_consumer) {
        // Old LFM2 IR only: Range -> Clamp (stand-in for Range -> Clamp -> Add -> Mod ->
        // Unsqueeze -> ScatterNDUpdate). The updated LFM2 IR does not have this consumer.
        auto clamp = std::make_shared<ov::op::v0::Clamp>(range, 0, 2);
        clamp->set_friendly_name("conv_clamp");
        auto clamp_result = std::make_shared<ov::op::v0::Result>(clamp);
        clamp_result->set_friendly_name("clamp_result");
        results.push_back(clamp_result);
    }

    return std::make_shared<ov::Model>(results, params, model_name);
}

// Updated LFM2 IR: RoPE + causal mask only. The Conv cache update no longer touches
// Range (it uses in-layer Slice + Broadcast instead), so Range has just two
// consumers.
std::shared_ptr<ov::Model> build_model_with_lfm2_v2_like_pattern() {
    return build_model_with_lfm2_like_pattern(/*offset_positions=*/false,
                                              /*with_range_clamp_consumer=*/false,
                                              "model_with_lfm2_v2_like_pattern");
}

// Gather-based (cache-table) RoPE, as in ONNX GroupQueryAttention-style exports. Instead of computing
// cos/sin from inv_freq*positions, precomputed cos/sin cache tables are indexed with the positions:
//
//   Range -> [Convert] -> Unsqueeze -> [Add(past_len)] -> Squeeze -> [Maximum] -> [Minimum] --> Gather (cos)
//                                                                                           --> Gather (sin)
//
// with_convert inserts the optional Convert right after Range; offset_positions inserts the optional
// past-length Add; with_clip inserts the optional Maximum/Minimum clip of the position into cache bounds.
// The cos and sin Gathers share the same position-producing Squeeze, matching the real export.
std::shared_ptr<ov::Model> build_model_with_gather_rope_pattern(bool with_convert = false,
                                                                bool offset_positions = false,
                                                                bool with_clip = true) {
    // Range: start=0, stop=seq_len, step=1  (mimics position_ids generation)
    const auto range_type = with_convert ? ov::element::i32 : ov::element::i64;
    auto start = ov::op::v0::Constant::create(range_type, ov::Shape{}, {0});
    auto stop = ov::op::v0::Constant::create(range_type, ov::Shape{}, {128});
    auto step = ov::op::v0::Constant::create(range_type, ov::Shape{}, {1});
    auto range = std::make_shared<ov::op::v4::Range>(start, stop, step, range_type);
    range->set_friendly_name("range");

    ov::Output<ov::Node> positions = range->output(0);
    if (with_convert) {
        auto convert = std::make_shared<ov::op::v0::Convert>(positions, ov::element::i64);
        convert->set_friendly_name("positions_convert");
        positions = convert->output(0);
    }

    // Unsqueeze: add batch dim [seq_len] → [1, seq_len]
    auto unsqueeze_axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
    auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(positions, unsqueeze_axes);
    unsqueeze->set_friendly_name("unsqueeze_batch");
    ov::Output<ov::Node> seq = unsqueeze->output(0);

    // transformers>=5.4: past length added after the Range
    if (offset_positions) {
        auto past_len = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {8});
        auto offset_add = std::make_shared<ov::op::v1::Add>(seq, past_len);
        offset_add->set_friendly_name("positions_offset_add");
        seq = offset_add->output(0);
    }

    // Squeeze: drop batch dim [1, seq_len] → [seq_len] to feed the 1-D Gather index
    auto squeeze_axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
    auto squeeze = std::make_shared<ov::op::v0::Squeeze>(seq, squeeze_axes);
    squeeze->set_friendly_name("positions_squeeze");
    ov::Output<ov::Node> indices = squeeze->output(0);

    // Optional clip of the absolute position into the cache bounds: Maximum(0) then Minimum(max_pos)
    if (with_clip) {
        auto lo = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
        auto clip_max = std::make_shared<ov::op::v1::Maximum>(indices, lo);
        clip_max->set_friendly_name("clip_max");
        auto hi = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {255});
        auto clip_min = std::make_shared<ov::op::v1::Minimum>(clip_max, hi);
        clip_min->set_friendly_name("clip_min");
        indices = clip_min->output(0);
    }

    // Precomputed cos/sin cache tables [max_pos, head_dim], indexed by the (shared) position sequence.
    auto gather_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
    auto cos_cache =
        ov::op::v0::Constant::create(ov::element::f32, ov::Shape{256, 8}, std::vector<float>(256 * 8, 1.0f));
    auto sin_cache =
        ov::op::v0::Constant::create(ov::element::f32, ov::Shape{256, 8}, std::vector<float>(256 * 8, 0.0f));
    auto gather_cos = std::make_shared<ov::op::v8::Gather>(cos_cache, indices, gather_axis);
    gather_cos->set_friendly_name("gather_cos");
    auto gather_sin = std::make_shared<ov::op::v8::Gather>(sin_cache, indices, gather_axis);
    gather_sin->set_friendly_name("gather_sin");

    auto cos_result = std::make_shared<ov::op::v0::Result>(gather_cos);
    cos_result->set_friendly_name("cos_result");
    auto sin_result = std::make_shared<ov::op::v0::Result>(gather_sin);
    sin_result->set_friendly_name("sin_result");

    return std::make_shared<ov::Model>(ov::ResultVector{cos_result, sin_result},
                                       ov::ParameterVector{},
                                       "model_with_gather_rope_pattern");
}

bool has_parameter_named(const std::shared_ptr<ov::Model>& model, const std::string& name) {
    for (const auto& p : model->get_parameters()) {
        for (const auto& n : p->output(0).get_names()) {
            if (n == name) {
                return true;
            }
        }
    }
    return false;
}

size_t count_parameters_named(const std::shared_ptr<ov::Model>& model, const std::string& name) {
    size_t count = 0;
    for (const auto& p : model->get_parameters()) {
        for (const auto& n : p->output(0).get_names()) {
            if (n == name) {
                ++count;
                break;
            }
        }
    }
    return count;
}

size_t count_ops_of_type(const std::shared_ptr<ov::Model>& model, const std::string& type_name) {
    size_t count = 0;
    for (const auto& op : model->get_ops()) {
        if (op->get_type_name() == type_name) {
            ++count;
        }
    }
    return count;
}

// Returns true if position_ids (through the new Squeeze and any preserved clip) feeds the given
// Gather's index input (input 1). Walks back along the data input (input 0) of clip/squeeze nodes.
bool gather_index_fed_by_position_ids(const std::shared_ptr<ov::Node>& gather) {
    auto walk = gather->input_value(1).get_node_shared_ptr();
    for (int depth = 0; depth < 6; ++depth) {
        if (walk->get_type_name() == std::string("Parameter")) {
            return walk->output(0).get_names().count("position_ids") > 0;
        }
        if (walk->get_input_size() == 0) {
            break;
        }
        walk = walk->input_value(0).get_node_shared_ptr();
    }
    return false;
}

// LFM3 has independent RoPE frequency branches, not just Cos/Sin sharing one Convert.
// Reuse the existing builder so each branch also has its own Range-based causal mask.
std::shared_ptr<ov::Model> build_model_with_independent_rope_patterns(bool offset_positions,
                                                                      bool with_range_clamp_consumer) {
    ov::ResultVector results;
    ov::ParameterVector params;
    for (size_t branch = 0; branch < 2; ++branch) {
        auto model = build_model_with_lfm2_like_pattern(offset_positions, with_range_clamp_consumer);
        const auto prefix = "branch_" + std::to_string(branch) + "_";
        for (const auto& op : model->get_ops()) {
            op->set_friendly_name(prefix + op->get_friendly_name());
            if (ov::is_type<ov::op::v1::LessEqual>(op)) {
                // A nonzero key index makes accidental mask rewiring numerically visible.
                op->input(0).replace_source_output(
                    ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1, 1, 1}, {64})->output(0));
            }
        }
        model->get_parameters().front()->output(0).set_names({prefix + "inv_freq"});
        results.insert(results.end(), model->get_results().begin(), model->get_results().end());
        params.insert(params.end(), model->get_parameters().begin(), model->get_parameters().end());
    }
    return std::make_shared<ov::Model>(results, params, "independent_rope_patterns");
}

class AddPositionIdsParamIndependentRopeTest : public testing::TestWithParam<std::tuple<bool, bool>> {
protected:
    void SetUp() override {
        model = build_model_with_independent_rope_patterns(std::get<0>(GetParam()), std::get<1>(GetParam()));
    }

    std::shared_ptr<ov::Model> model;
};

TEST_P(AddPositionIdsParamIndependentRopeTest, SharesPositionIdsAcrossBothBranchesAndPreservesMasks) {
    ov::NodeVector converts;
    ov::NodeVector clamps;
    std::set<ov::Node*> mask_ranges;
    std::vector<std::pair<ov::Input<ov::Node>, ov::Output<ov::Node>>> mask_inputs;
    for (const auto& op : model->get_ops()) {
        if (ov::is_type<ov::op::v0::Convert>(op)) {
            converts.push_back(op);
        } else if (ov::is_type<ov::op::v0::Clamp>(op)) {
            clamps.push_back(op);
        } else if (ov::is_type<ov::op::v1::LessEqual>(op)) {
            // Snapshot every data edge from this mask to its original Range, including Add.
            mask_inputs.emplace_back(op->input(1), op->input_value(1));
            auto node = op->input_value(1).get_node_shared_ptr();
            while (!ov::is_type<ov::op::v4::Range>(node)) {
                ASSERT_GT(node->get_input_size(), 0u);
                mask_inputs.emplace_back(node->input(0), node->input_value(0));
                node = node->input_value(0).get_node_shared_ptr();
            }
            mask_ranges.insert(node.get());
        }
    }
    ASSERT_EQ(converts.size(), 2u);
    ASSERT_EQ(mask_ranges.size(), 2u);
    ASSERT_EQ(clamps.size(), std::get<1>(GetParam()) ? 2u : 0u);

    ASSERT_NO_THROW(ov::npuw::AddPositionIdsParam().run_on_model(model));
    ASSERT_EQ(count_parameters_named(model, "position_ids"), 1u);
    ASSERT_EQ(model->get_parameters().size(), 3u);  // Two frequency tables, one position_ids.
    std::shared_ptr<ov::op::v0::Parameter> position_ids;
    for (const auto& param : model->get_parameters()) {
        if (param->output(0).get_names().count("position_ids")) {
            position_ids = param;
        }
    }
    ASSERT_NE(position_ids, nullptr);
    EXPECT_EQ(position_ids->get_element_type(), ov::element::i64);
    EXPECT_EQ(position_ids->get_partial_shape(), (ov::PartialShape{-1, -1}));
    for (const auto& convert : converts) {
        SCOPED_TRACE(convert->get_friendly_name());
        auto parent = convert->input_value(0).get_node_shared_ptr();
        ASSERT_TRUE(ov::is_type<ov::op::v0::Unsqueeze>(parent));
        EXPECT_EQ(parent->input_value(0), position_ids->output(0));
    }
    for (const auto& edge : mask_inputs) {
        SCOPED_TRACE(edge.first.get_node()->get_friendly_name());
        EXPECT_EQ(edge.first.get_source_output(), edge.second);
    }
    EXPECT_EQ(count_ops_of_type(model, "Range"), 2u);
    for (const auto& clamp : clamps) {
        SCOPED_TRACE(clamp->get_friendly_name());
        auto parent = clamp->input_value(0).get_node_shared_ptr();
        ASSERT_TRUE(ov::is_type<ov::op::v0::Squeeze>(parent));
        EXPECT_EQ(parent->input_value(0), position_ids->output(0));
    }

    const auto ops_after_first = model->get_ops().size();
    ASSERT_NO_THROW(ov::npuw::AddPositionIdsParam().run_on_model(model));
    EXPECT_EQ(model->get_parameters().size(), 3u);
    EXPECT_EQ(count_parameters_named(model, "position_ids"), 1u);
    EXPECT_EQ(model->get_ops().size(), ops_after_first);
}

TEST_P(AddPositionIdsParamIndependentRopeTest, EvaluatesBothFrequencyTablesUsingExternalPositions) {
    ASSERT_NO_THROW(ov::npuw::AddPositionIdsParam().run_on_model(model));
    ASSERT_EQ(count_parameters_named(model, "position_ids"), 1u);
    ASSERT_EQ(model->get_parameters().size(), 3u);

    ov::TensorVector inputs;
    for (const auto& param : model->get_parameters()) {
        const auto& names = param->output(0).get_names();
        if (names.count("position_ids")) {
            ov::Tensor positions(ov::element::i64, ov::Shape{1, 128});
            for (size_t i = 0; i < 128; ++i) {
                // Padding and nonconsecutive absolute positions differ from both Range and Range + 8.
                positions.data<int64_t>()[i] = i % 5 == 0 ? 0 : 37 + 2 * static_cast<int64_t>(i);
            }
            inputs.push_back(positions);
        } else {
            ASSERT_TRUE(names.count("branch_0_inv_freq") || names.count("branch_1_inv_freq"));
            const float scale = names.count("branch_0_inv_freq") ? 0.01f : 0.03f;
            ov::Tensor frequencies(ov::element::f32, ov::Shape{1, 8, 1});
            for (size_t i = 0; i < 8; ++i) {
                frequencies.data<float>()[i] = scale * static_cast<float>(i + 1);
            }
            inputs.push_back(frequencies);
        }
    }
    ov::TensorVector outputs;
    for (const auto& result : model->get_results()) {
        outputs.emplace_back(result->get_element_type(), ov::Shape{});
    }
    ASSERT_TRUE(model->evaluate(outputs, inputs));

    const size_t results_per_branch = std::get<1>(GetParam()) ? 4 : 3;
    for (size_t branch = 0; branch < 2; ++branch) {
        SCOPED_TRACE(branch);
        const auto& cos = outputs[branch * results_per_branch];
        const auto& sin = outputs[branch * results_per_branch + 1];
        const auto& mask = outputs[branch * results_per_branch + 2];
        ASSERT_EQ(cos.get_shape(), (ov::Shape{1, 128, 16}));
        ASSERT_EQ(sin.get_shape(), cos.get_shape());
        ASSERT_EQ(mask.get_shape(), (ov::Shape{1, 1, 1, 128}));
        const float scale = branch == 0 ? 0.01f : 0.03f;
        for (size_t i = 0; i < 128; ++i) {
            SCOPED_TRACE(i);
            const int64_t position = i % 5 == 0 ? 0 : 37 + 2 * static_cast<int64_t>(i);
            for (size_t j = 0; j < 16; ++j) {
                const float angle = (scale * static_cast<float>(j % 8 + 1)) * static_cast<float>(position);
                EXPECT_NEAR(cos.data<const float>()[i * 16 + j], std::cos(angle), 1e-5f);
                EXPECT_NEAR(sin.data<const float>()[i * 16 + j], std::sin(angle), 1e-5f);
            }
            const auto original_position = static_cast<int64_t>(i) + (std::get<0>(GetParam()) ? 8 : 0);
            EXPECT_EQ(mask.data<const char>()[i] != 0, 64 <= original_position);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(OffsetAndLegacyClamp,
                         AddPositionIdsParamIndependentRopeTest,
                         testing::Combine(testing::Bool(), testing::Bool()));

// ===================== TESTS =====================
TEST(AddPositionIdsParamTest, AddsPositionIdsParameter) {
    auto model = build_model_with_lfm2_like_pattern();

    EXPECT_FALSE(has_parameter_named(model, "position_ids"));
    ASSERT_NO_THROW(ov::npuw::AddPositionIdsParam().run_on_model(model));
    EXPECT_TRUE(has_parameter_named(model, "position_ids"));
}

TEST(AddPositionIdsParamTest, PositionIdsHasCorrectShapeAndType) {
    auto model = build_model_with_lfm2_like_pattern();
    ov::npuw::AddPositionIdsParam().run_on_model(model);

    for (const auto& p : model->get_parameters()) {
        for (const auto& n : p->output(0).get_names()) {
            if (n == "position_ids") {
                EXPECT_EQ(p->get_element_type(), ov::element::i64);
                const auto& shape = p->get_partial_shape();
                ASSERT_EQ(shape.rank().get_length(), 2);
                EXPECT_TRUE(shape[0].is_dynamic());
                EXPECT_TRUE(shape[1].is_dynamic());
                return;
            }
        }
    }
    FAIL() << "position_ids parameter not found";
}

// --- Test: RoPE path uses position_ids, not Range ---
// After the pass, the MatMul in the RoPE path should ultimately be fed by position_ids
// (through the new Unsqueeze), not by the original Range.
TEST(AddPositionIdsParamTest, RopePathUsesPositionIds) {
    auto model = build_model_with_lfm2_like_pattern();
    ov::npuw::AddPositionIdsParam().run_on_model(model);

    // Find the MatMul node
    std::shared_ptr<ov::Node> matmul_node;
    for (const auto& op : model->get_ops()) {
        if (op->get_type_name() == std::string("MatMul")) {
            matmul_node = op;
            break;
        }
    }
    ASSERT_NE(matmul_node, nullptr) << "MatMul not found in model";

    // Walk backwards from MatMul's input 1 to find what feeds it
    // The chain should be: position_ids → Unsqueeze → Convert → MatMul
    auto walk = matmul_node->input_value(1).get_node_shared_ptr();
    bool found_position_ids = false;
    int depth = 0;
    while (depth < 5) {  // limit walk depth
        if (walk->get_type_name() == std::string("Parameter")) {
            const auto& names = walk->output(0).get_names();
            if (names.count("position_ids") > 0) {
                found_position_ids = true;
            }
            break;
        }
        if (walk->get_input_size() == 0) {
            break;
        }
        walk = walk->input_value(0).get_node_shared_ptr();
        ++depth;
    }
    EXPECT_TRUE(found_position_ids) << "MatMul (RoPE path) should be fed by position_ids parameter";
}

TEST(AddPositionIdsParamTest, RangePreservedForCausalMaskConsumer) {
    auto model = build_model_with_lfm2_like_pattern();
    ov::npuw::AddPositionIdsParam().run_on_model(model);

    // Range should still exist in the graph
    EXPECT_GE(count_ops_of_type(model, "Range"), 1u) << "Range node should be preserved for causal mask";

    // Find the LessEqual node -- Range should still feed it via the Unsqueeze chain
    bool found_range_as_le_input = false;
    for (const auto& op : model->get_ops()) {
        if (op->get_type_name() == std::string("LessEqual")) {
            // BFS from all LessEqual inputs to find Range
            std::set<ov::Node*> visited;
            std::queue<ov::Node*> to_visit;
            for (size_t inp = 0; inp < op->get_input_size(); ++inp) {
                to_visit.push(op->input_value(inp).get_node());
            }
            while (!to_visit.empty()) {
                auto* n = to_visit.front();
                to_visit.pop();
                if (!visited.insert(n).second) {
                    continue;
                }
                if (std::string(n->get_type_name()) == "Range") {
                    found_range_as_le_input = true;
                    break;
                }
                for (size_t i = 0; i < n->get_input_size(); ++i) {
                    to_visit.push(n->input_value(i).get_node());
                }
            }
        }
    }
    EXPECT_TRUE(found_range_as_le_input) << "Range should still feed the causal mask LessEqual";
}

TEST(AddPositionIdsParamTest, NoOpWhenPatternDoesNotMatch) {
    // Build a model without the RoPE pattern
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, 64});
    input->output(0).set_names({"input"});
    auto result = std::make_shared<ov::op::v0::Result>(input);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input}, "no_rope_model");

    size_t params_before = model->get_parameters().size();

    ASSERT_NO_THROW(ov::npuw::AddPositionIdsParam().run_on_model(model));

    EXPECT_EQ(model->get_parameters().size(), params_before)
        << "No new parameters should be added when pattern doesn't match";
}

// --- Test: Clamp consumer on Range gets rewired to position_ids ---
// In old LFM2 models, Range feeds a Clamp for Gated Short Convolution indexing.
// The pass should replace Range->Clamp with Squeeze(position_ids)->Clamp.
TEST(AddPositionIdsParamTest, ClampInputIsReplacedToPositionIds) {
    auto model = build_model_with_lfm2_like_pattern();

    ASSERT_NO_THROW(ov::npuw::AddPositionIdsParam().run_on_model(model));
    EXPECT_TRUE(has_parameter_named(model, "position_ids"));

    // Clamp's input should now come from Squeeze(position_ids), not Range
    for (const auto& op : model->get_ops()) {
        if (op->get_type_name() != std::string("Clamp")) {
            continue;
        }
        auto producer = op->input_value(0).get_node_shared_ptr();
        ASSERT_EQ(std::string(producer->get_type_name()), "Squeeze")
            << "Clamp should be fed by Squeeze(position_ids), not " << producer->get_type_name();

        auto squeeze_input = producer->input_value(0).get_node_shared_ptr();
        ASSERT_EQ(std::string(squeeze_input->get_type_name()), "Parameter");
        EXPECT_TRUE(squeeze_input->output(0).get_names().count("position_ids") > 0)
            << "Squeeze should be fed by position_ids parameter";
    }
}

// Running the pass a second time must not alter the graph (no duplicate position_ids).
TEST(AddPositionIdsParamTest, ReapplyDoesNotModifyGraph) {
    auto model = build_model_with_lfm2_like_pattern();
    ov::npuw::AddPositionIdsParam().run_on_model(model);

    const size_t params_after_first = model->get_parameters().size();
    const size_t ops_after_first = model->get_ops().size();

    ov::npuw::AddPositionIdsParam().run_on_model(model);

    EXPECT_EQ(model->get_parameters().size(), params_after_first)
        << "Second pass should not add duplicate parameters";
    EXPECT_EQ(model->get_ops().size(), ops_after_first)
        << "Second pass should not add duplicate operations";
}

// ===================== LFM2 v2 (updated IR) tests =====================
// The updated LFM2 IR no longer routes `Range` through the Range -> Clamp -> Add ->
// Mod -> Unsqueeze -> ScatterNDUpdate path. The Conv cache is now updated in-layer
// via Slice(last 3) + Broadcast, entirely decoupled from `Range`.
// Therefore `Range` has only two consumers:
//   Branch A: Range -> Unsqueeze -> Unsqueeze -> Convert -> MatMul  (RoPE)
//   Branch B: Range -> Unsqueeze -> Unsqueeze -> Unsqueeze -> LessEqual  (causal mask)
// The AddPositionIdsParam pass must still add `position_ids`, rewire the RoPE branch
// to it, and preserve `Range` for the causal-mask branch.

// After the pass: `position_ids` parameter is added with i64 / {?, ?}.
TEST(AddPositionIdsParamTest, LFM2v2_AddsPositionIdsParameter) {
    auto model = build_model_with_lfm2_v2_like_pattern();

    EXPECT_FALSE(has_parameter_named(model, "position_ids"));
    ASSERT_NO_THROW(ov::npuw::AddPositionIdsParam().run_on_model(model));
    EXPECT_TRUE(has_parameter_named(model, "position_ids"));

    for (const auto& p : model->get_parameters()) {
        for (const auto& n : p->output(0).get_names()) {
            if (n == "position_ids") {
                EXPECT_EQ(p->get_element_type(), ov::element::i64);
                const auto& shape = p->get_partial_shape();
                ASSERT_EQ(shape.rank().get_length(), 2);
                EXPECT_TRUE(shape[0].is_dynamic());
                EXPECT_TRUE(shape[1].is_dynamic());
                return;
            }
        }
    }
    FAIL() << "position_ids parameter not found";
}

// After the pass: the RoPE branch (MatMul) is fed by position_ids, not by Range.
TEST(AddPositionIdsParamTest, LFM2v2_RopePathUsesPositionIds) {
    auto model = build_model_with_lfm2_v2_like_pattern();
    ov::npuw::AddPositionIdsParam().run_on_model(model);

    std::shared_ptr<ov::Node> matmul_node;
    for (const auto& op : model->get_ops()) {
        if (op->get_type_name() == std::string("MatMul")) {
            matmul_node = op;
            break;
        }
    }
    ASSERT_NE(matmul_node, nullptr) << "MatMul not found in model";

    // Walk backwards from MatMul's input 1: position_ids -> Unsqueeze -> Convert -> MatMul.
    auto walk = matmul_node->input_value(1).get_node_shared_ptr();
    bool found_position_ids = false;
    int depth = 0;
    while (depth < 5) {
        if (walk->get_type_name() == std::string("Parameter")) {
            const auto& names = walk->output(0).get_names();
            if (names.count("position_ids") > 0) {
                found_position_ids = true;
            }
            break;
        }
        if (walk->get_input_size() == 0) {
            break;
        }
        walk = walk->input_value(0).get_node_shared_ptr();
        ++depth;
    }
    EXPECT_TRUE(found_position_ids) << "MatMul (RoPE path) should be fed by position_ids parameter";
}

// After the pass: Range is still present and still feeds the causal-mask LessEqual.
TEST(AddPositionIdsParamTest, LFM2v2_RangePreservedForCausalMask) {
    auto model = build_model_with_lfm2_v2_like_pattern();
    ov::npuw::AddPositionIdsParam().run_on_model(model);

    EXPECT_GE(count_ops_of_type(model, "Range"), 1u) << "Range node should be preserved for causal mask";

    bool found_range_as_le_input = false;
    for (const auto& op : model->get_ops()) {
        if (op->get_type_name() != std::string("LessEqual")) {
            continue;
        }
        std::set<ov::Node*> visited;
        std::queue<ov::Node*> to_visit;
        for (size_t inp = 0; inp < op->get_input_size(); ++inp) {
            to_visit.push(op->input_value(inp).get_node());
        }
        while (!to_visit.empty()) {
            auto* n = to_visit.front();
            to_visit.pop();
            if (!visited.insert(n).second) {
                continue;
            }
            if (std::string(n->get_type_name()) == "Range") {
                found_range_as_le_input = true;
                break;
            }
            for (size_t i = 0; i < n->get_input_size(); ++i) {
                to_visit.push(n->input_value(i).get_node());
            }
        }
    }
    EXPECT_TRUE(found_range_as_le_input) << "Range should still feed the causal mask LessEqual";
}


// --- transformers>=5.4: Range -> Add(past_len) -> Unsqueeze -> Unsqueeze -> Convert -> RoPE ---
// The offset moved out of Range, so the pass matches through an optional Add. These models also
// give the causal mask its own Range and have no Clamp path.
TEST(AddPositionIdsParamTest, OffsetAddRopePathUsesPositionIds) {
    auto model = build_model_with_lfm2_like_pattern(true /*offset_positions*/, false /*with_range_clamp_consumer*/);

    EXPECT_FALSE(has_parameter_named(model, "position_ids"));
    ASSERT_NO_THROW(ov::npuw::AddPositionIdsParam().run_on_model(model));
    EXPECT_TRUE(has_parameter_named(model, "position_ids"));

    // Walk back from MatMul: position_ids → Unsqueeze → Convert → MatMul, no Add in between
    std::shared_ptr<ov::Node> matmul_node;
    for (const auto& op : model->get_ops()) {
        if (op->get_type_name() == std::string("MatMul")) {
            matmul_node = op;
            break;
        }
    }
    ASSERT_NE(matmul_node, nullptr) << "MatMul not found in model";

    auto walk = matmul_node->input_value(1).get_node_shared_ptr();
    bool found_position_ids = false;
    for (int depth = 0; depth < 5; ++depth) {  // limit walk depth
        if (walk->get_type_name() == std::string("Parameter")) {
            found_position_ids = walk->output(0).get_names().count("position_ids") > 0;
            break;
        }
        if (walk->get_input_size() == 0) {
            break;
        }
        walk = walk->input_value(0).get_node_shared_ptr();
    }
    EXPECT_TRUE(found_position_ids) << "MatMul (RoPE path) should be fed by position_ids, not Add(Range, past_len)";
}

// The Range must survive for Causal Mask creation even though the RoPE consumer it used to be
// identified by is now the Add rather than the Unsqueeze.
TEST(AddPositionIdsParamTest, OffsetAddPreservesRangeForCausalMask) {
    auto model = build_model_with_lfm2_like_pattern(true /*offset_positions*/, false /*with_range_clamp_consumer*/);
    ASSERT_NO_THROW(ov::npuw::AddPositionIdsParam().run_on_model(model));

    EXPECT_GE(count_ops_of_type(model, "Range"), 1u) << "Range node should be preserved for causal mask";

    bool found_range_as_le_input = false;
    for (const auto& op : model->get_ops()) {
        if (op->get_type_name() != std::string("LessEqual")) {
            continue;
        }
        std::set<ov::Node*> visited;
        std::queue<ov::Node*> to_visit;
        for (size_t inp = 0; inp < op->get_input_size(); ++inp) {
            to_visit.push(op->input_value(inp).get_node());
        }
        while (!to_visit.empty()) {
            auto* n = to_visit.front();
            to_visit.pop();
            if (!visited.insert(n).second) {
                continue;
            }
            if (std::string(n->get_type_name()) == "Range") {
                found_range_as_le_input = true;
                break;
            }
            for (size_t i = 0; i < n->get_input_size(); ++i) {
                to_visit.push(n->input_value(i).get_node());
            }
        }
    }
    EXPECT_TRUE(found_range_as_le_input) << "Range should still feed the causal mask LessEqual";
}

// Clamp rewiring now keys off the Clamp itself rather than off skipping the RoPE consumer, so it
// has to keep working when the positions reach RoPE through an Add.
TEST(AddPositionIdsParamTest, OffsetAddWithClampReplacesClampInput) {
    auto model = build_model_with_lfm2_like_pattern(true /*offset_positions*/, true /*with_range_clamp_consumer*/);

    ASSERT_NO_THROW(ov::npuw::AddPositionIdsParam().run_on_model(model));
    EXPECT_TRUE(has_parameter_named(model, "position_ids"));
    ASSERT_EQ(count_ops_of_type(model, "Clamp"), 1u) << "Test model should have a Clamp to rewire";

    for (const auto& op : model->get_ops()) {
        if (op->get_type_name() != std::string("Clamp")) {
            continue;
        }
        auto producer = op->input_value(0).get_node_shared_ptr();
        ASSERT_EQ(std::string(producer->get_type_name()), "Squeeze")
            << "Clamp should be fed by Squeeze(position_ids), not " << producer->get_type_name();

        auto squeeze_input = producer->input_value(0).get_node_shared_ptr();
        ASSERT_EQ(std::string(squeeze_input->get_type_name()), "Parameter");
        EXPECT_TRUE(squeeze_input->output(0).get_names().count("position_ids") > 0)
            << "Squeeze should be fed by position_ids parameter";
    }
}

// ===================== GATHER-BASED (CACHE-TABLE) RoPE TESTS =====================
// ONNX GroupQueryAttention-style exports index precomputed cos/sin cache tables with the positions,
// so the position sequence drives a Gather rather than a Cos/Sin. The pass must still synthesize a
// position_ids parameter for these models.

TEST(AddPositionIdsParamTest, GatherRopeAddsPositionIdsParameter) {
    auto model = build_model_with_gather_rope_pattern();

    EXPECT_FALSE(has_parameter_named(model, "position_ids"));
    ASSERT_NO_THROW(ov::npuw::AddPositionIdsParam().run_on_model(model));
    EXPECT_TRUE(has_parameter_named(model, "position_ids"));
}

// Both the cos-cache and sin-cache Gathers share one position-producing Squeeze, so exactly one
// position_ids parameter must be created (dedup), not one per Gather.
TEST(AddPositionIdsParamTest, GatherRopeCreatesSinglePositionIdsForCosAndSin) {
    auto model = build_model_with_gather_rope_pattern();
    ov::npuw::AddPositionIdsParam().run_on_model(model);

    EXPECT_EQ(count_parameters_named(model, "position_ids"), 1u)
        << "The shared cos/sin Gathers should yield exactly one position_ids parameter";
}

// After the pass, both Gathers must be indexed by position_ids (through the preserved clip), not by
// the original Range/Squeeze chain.
TEST(AddPositionIdsParamTest, GatherRopeGatherIndexUsesPositionIds) {
    auto model = build_model_with_gather_rope_pattern();
    ov::npuw::AddPositionIdsParam().run_on_model(model);

    size_t gathers_checked = 0;
    for (const auto& op : model->get_ops()) {
        if (op->get_type_name() != std::string("Gather")) {
            continue;
        }
        ++gathers_checked;
        EXPECT_TRUE(gather_index_fed_by_position_ids(op))
            << "Gather '" << op->get_friendly_name() << "' should be indexed by position_ids";
    }
    EXPECT_EQ(gathers_checked, 2u) << "Model should contain both the cos-cache and sin-cache Gathers";
}

// The optional Convert / Add(past_len) / clip nodes are all absent in some exports; the pass must
// match through the bare Range -> Unsqueeze -> Squeeze -> Gather chain too.
TEST(AddPositionIdsParamTest, GatherRopeMatchesWithoutOptionalNodes) {
    auto model = build_model_with_gather_rope_pattern(false /*with_convert*/, false /*offset*/, false /*with_clip*/);

    EXPECT_FALSE(has_parameter_named(model, "position_ids"));
    ASSERT_NO_THROW(ov::npuw::AddPositionIdsParam().run_on_model(model));
    EXPECT_TRUE(has_parameter_named(model, "position_ids"));
    EXPECT_EQ(count_parameters_named(model, "position_ids"), 1u);

    for (const auto& op : model->get_ops()) {
        if (op->get_type_name() == std::string("Gather")) {
            EXPECT_TRUE(gather_index_fed_by_position_ids(op))
                << "Gather should be indexed by position_ids even without clip/convert/add";
        }
    }
}

// The optional Convert (after Range) and Add(past_len) must not prevent the match.
TEST(AddPositionIdsParamTest, GatherRopeMatchesWithConvertAndOffsetAdd) {
    auto model = build_model_with_gather_rope_pattern(true /*with_convert*/, true /*offset*/, true /*with_clip*/);

    EXPECT_FALSE(has_parameter_named(model, "position_ids"));
    ASSERT_NO_THROW(ov::npuw::AddPositionIdsParam().run_on_model(model));
    EXPECT_TRUE(has_parameter_named(model, "position_ids"));
    EXPECT_EQ(count_parameters_named(model, "position_ids"), 1u);

    for (const auto& op : model->get_ops()) {
        if (op->get_type_name() == std::string("Gather")) {
            EXPECT_TRUE(gather_index_fed_by_position_ids(op))
                << "Gather should be indexed by position_ids through the optional Convert/Add";
        }
    }
}

// Reapplying the pass to an already-transformed Gather-RoPE model must be a no-op.
TEST(AddPositionIdsParamTest, GatherRopeReapplyDoesNotModifyGraph) {
    auto model = build_model_with_gather_rope_pattern();
    ov::npuw::AddPositionIdsParam().run_on_model(model);

    const size_t params_after_first = model->get_parameters().size();
    const size_t ops_after_first = model->get_ops().size();

    ov::npuw::AddPositionIdsParam().run_on_model(model);

    EXPECT_EQ(model->get_parameters().size(), params_after_first) << "Second pass should not add duplicate parameters";
    EXPECT_EQ(model->get_ops().size(), ops_after_first) << "Second pass should not add duplicate operations";
}
}  // namespace
