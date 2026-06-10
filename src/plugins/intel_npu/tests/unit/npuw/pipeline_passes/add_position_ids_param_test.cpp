// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <queue>
#include <set>
#include <string>
#include <vector>

#include "npuw_transformations/add_position_ids_param.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/less_eq.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"

namespace {
std::shared_ptr<ov::Model> build_model_with_lfm2_like_pattern() {
    // Range: start=0, stop=seq_len, step=1  (mimics position_ids generation)
    auto start = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
    auto stop = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {128});
    auto step = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
    auto range = std::make_shared<ov::op::v4::Range>(start, stop, step, ov::element::i64);
    range->set_friendly_name("range");

    // Unsqueeze: add batch dim [seq_len] → [1, seq_len]
    auto unsqueeze_axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
    auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(range, unsqueeze_axes);
    unsqueeze->set_friendly_name("unsqueeze_batch");

    // Unsqueeze1: add feature dim [1, seq_len] → [1, 1, seq_len]
    auto unsqueeze1_axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    auto unsqueeze1 = std::make_shared<ov::op::v0::Unsqueeze>(unsqueeze, unsqueeze1_axes);
    unsqueeze1->set_friendly_name("unsqueeze_feature");

    // Convert (always present in real LFM-2 models)
    auto convert = std::make_shared<ov::op::v0::Convert>(unsqueeze1, ov::element::f32);
    convert->set_friendly_name("convert");

    // MatMul: [inv_freq] × [positions] → freqs
    auto inv_freq = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 8, 1});
    inv_freq->output(0).set_names({"inv_freq"});
    inv_freq->set_friendly_name("inv_freq");

    auto matmul = std::make_shared<ov::op::v0::MatMul>(inv_freq, convert);
    matmul->set_friendly_name("matmul_rope");

    // Transpose: [1, 8, 128] → [1, 128, 8]
    auto transpose_order = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {0, 2, 1});
    auto transpose = std::make_shared<ov::op::v1::Transpose>(matmul, transpose_order);
    transpose->set_friendly_name("transpose_rope");

    // Concat(transpose, transpose) → simulate theta doubling
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

    // Optional Clamp consumer on Range (simulates Gated Short Convolution indexing in LFM2).
    // Real LFM2 path: Range -> Clamp -> Add -> Mod -> Unsqueeze -> ScatterNDUpdate
    auto clamp = std::make_shared<ov::op::v0::Clamp>(range, 0, 2);
    clamp->set_friendly_name("conv_clamp");
    auto clamp_result = std::make_shared<ov::op::v0::Result>(clamp);
    clamp_result->set_friendly_name("clamp_result");
    results.push_back(clamp_result);

    return std::make_shared<ov::Model>(results, params, "model_with_lfm2_like_pattern");
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

size_t count_ops_of_type(const std::shared_ptr<ov::Model>& model, const std::string& type_name) {
    size_t count = 0;
    for (const auto& op : model->get_ops()) {
        if (op->get_type_name() == type_name) {
            ++count;
        }
    }
    return count;
}

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
// In real LFM2 models, Range feeds a Clamp for Gated Short Convolution indexing.
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
}  // namespace
