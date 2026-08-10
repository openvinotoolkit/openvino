// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "npuw_transformations/remove_token_type_ids.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <numeric>
#include <set>
#include <string>
#include <vector>

#include "openvino/op/ops.hpp"

// Tests for ov::npuw::RemoveTokenTypeIds.
//
// The pass detaches two independent groups of `token_type_ids` (TTI) consumers from a Gemma-3
// generate model and then drops the parameter itself:
//
// 1. RemoveTTIVisionSubgraph - the blockwise vision mask. Two chains start at TTI and meet in
//    `BitwiseAnd`, which is OR-ed into the causal (or sliding causal) mask, once per attention
//    branch (global and local):
//
//      subg1: TTI -> Equal -> Pad -> Slice -> BitwiseNot -> BitwiseAnd -> Convert -> CumSum
//                 -> Add -> Convert -> Select -> ShapeOf -> Gather -> Less
//                 -> per branch: Select -> Equal
//      subg2: TTI -> Reshape -> per branch: two Gather -> Reshape -> Reshape -> Equal chains
//                 (only the second one carries a Select) -> BitwiseAnd
//
//      vision_block = BitwiseAnd(subg2_branch, subg1_branch)
//      merge point  = BitwiseOr(causal_mask, vision_block)
//
//    The callback re-points each `BitwiseOr` input[1] at `Constant(false)`, which leaves the causal
//    mask untouched and drops the last consumer of the whole vision subgraph.
//
// 2. RemoveTTIShapeOfSubgraph - a ShapeOf chain that supplies the target shape for a `Reshape`
//    inside the `attention_mask` chain:
//
//      TTI -> ShapeOf -> Gather -> Less -> Select -> Convert -> Add -> ShapeOf ---+
//      attention_mask -> Convert -> Reshape -> Gather -> Reshape -> Reshape <-----+
//
//    Here the subgraph cannot simply be cut, so the callback re-points the second ShapeOf at the
//    `Gather` indices instead - which is what a model without TTI reshapes against. The
//    `attention_mask` chain itself must come out untouched.
//
// Removal contract: the parameter is dropped if and only if nothing reads it anymore. Either pattern
// firing on its own is enough, as long as it accounted for every consumer. If a consumer survives the
// rewrites the pass throws, because by then the graph is half-transformed and returning it would
// silently corrupt the generated output.

namespace {

using namespace ov;
using namespace ov::op;

constexpr size_t SEQ = 8;
constexpr int64_t SEQ_I64 = static_cast<int64_t>(SEQ);

std::shared_ptr<v0::Parameter> make_token_type_ids() {
    auto tti = std::make_shared<v0::Parameter>(element::i64, Shape{1, SEQ});
    tti->set_friendly_name("token_type_ids");
    tti->output(0).set_names({"token_type_ids"});
    return tti;
}

std::shared_ptr<v0::Parameter> make_named_param(const element::Type& type,
                                                const Shape& shape,
                                                const std::string& name) {
    auto param = std::make_shared<v0::Parameter>(type, shape);
    param->set_friendly_name(name);
    param->output(0).set_names({name});
    return param;
}

// Appends the vision-mask subgraph matched by RemoveTTIVisionSubgraph: one shared blockwise-mask
// chain (subg1) and one shared Reshape chain (subg2), both branching into a global and a local
// attention path that each merge into BitwiseOr(causal_mask, vision_block).
void add_vision_chain(const std::shared_ptr<v0::Parameter>& tti, ParameterVector& params, ResultVector& results) {
    const auto zeros = v0::Constant::create(element::i64, Shape{}, {0});

    // ---- subg1: blockwise mask, shared by both branches up to `Less` ----
    auto subg1_equal = std::make_shared<v1::Equal>(tti, v0::Constant::create(element::i64, Shape{1, 1}, {1}));
    auto subg1_pad = std::make_shared<v12::Pad>(subg1_equal,
                                                v0::Constant::create(element::i64, Shape{2}, {0, 1}),
                                                v0::Constant::create(element::i64, Shape{2}, {0, 0}),
                                                v0::Constant::create(element::boolean, Shape{}, {false}),
                                                PadMode::CONSTANT);
    auto subg1_slice = std::make_shared<v8::Slice>(subg1_pad,
                                                   v0::Constant::create(element::i64, Shape{1}, {0}),
                                                   v0::Constant::create(element::i64, Shape{1}, {SEQ}),
                                                   v0::Constant::create(element::i64, Shape{1}, {1}),
                                                   v0::Constant::create(element::i64, Shape{1}, {1}));
    auto subg1_bw_not = std::make_shared<v13::BitwiseNot>(subg1_slice);
    auto subg1_bw_and = std::make_shared<v13::BitwiseAnd>(subg1_equal, subg1_bw_not);
    auto subg1_convert = std::make_shared<v0::Convert>(subg1_bw_and, element::i32);
    auto subg1_cumsum = std::make_shared<v0::CumSum>(subg1_convert, v0::Constant::create(element::i64, Shape{}, {1}));
    auto subg1_add = std::make_shared<v1::Add>(subg1_cumsum, v0::Constant::create(element::i32, Shape{1, 1}, {-1}));
    auto subg1_convert_add = std::make_shared<v0::Convert>(subg1_add, element::i64);
    auto subg1_select = std::make_shared<v1::Select>(subg1_equal, subg1_convert_add, zeros);
    auto subg1_shape_of = std::make_shared<v3::ShapeOf>(subg1_select, element::i64);
    auto subg1_gather = std::make_shared<v8::Gather>(subg1_shape_of,
                                                     v0::Constant::create(element::i64, Shape{}, {1}),
                                                     v0::Constant::create(element::i32, Shape{}, {0}));

    auto range_input = make_named_param(element::i64, Shape{1, 1, 1, SEQ}, "range_input");
    auto subg1_less = std::make_shared<v1::Less>(range_input, subg1_gather);
    params.push_back(range_input);

    // ---- subg2: Reshape of `token_type_ids`, shared by both branches ----
    auto subg2_reshape = std::make_shared<v1::Reshape>(tti, v0::Constant::create(element::i64, Shape{1}, {SEQ}), false);

    std::vector<int64_t> iota_values(SEQ);
    std::iota(iota_values.begin(), iota_values.end(), 0);

    const auto row_shape = v0::Constant::create(element::i64, Shape{2}, {int64_t{1}, SEQ_I64});
    const auto col_shape = v0::Constant::create(element::i64, Shape{4}, {int64_t{1}, int64_t{1}, int64_t{1}, SEQ_I64});
    const auto group_ids = v0::Constant::create(element::i64, Shape{1, 1, SEQ, 1}, iota_values);
    const auto gather_indices = v0::Constant::create(element::i64, Shape{SEQ}, iota_values);
    const auto gather_axis = v0::Constant::create(element::i32, Shape{}, {0});

    // ---- one path per attention branch ----
    for (const char* branch : {"global", "local"}) {
        const std::string suffix = std::string("_") + branch;

        auto select_input = make_named_param(element::i64, Shape{1, 1, 1, SEQ}, "select_input" + suffix);
        auto branch_select = std::make_shared<v1::Select>(subg1_less, select_input, zeros);
        auto image_group_ids = make_named_param(element::i64, Shape{1, 1, SEQ, 1}, "image_group_ids" + suffix);
        auto branch_equal = std::make_shared<v1::Equal>(image_group_ids, branch_select);

        // First subg2 chain: Gather -> Reshape -> Reshape -> Equal.
        auto gather1 = std::make_shared<v8::Gather>(subg2_reshape, gather_indices, gather_axis);
        auto gather1_reshape = std::make_shared<v1::Reshape>(gather1, row_shape, false);
        auto gather1_reshape2 = std::make_shared<v1::Reshape>(gather1_reshape, col_shape, false);
        auto gather1_equal = std::make_shared<v1::Equal>(gather1_reshape2, group_ids);

        // Second subg2 chain: same, but with a `Select` before the `Equal`.
        auto gather2 = std::make_shared<v8::Gather>(subg2_reshape, gather_indices, gather_axis);
        auto gather2_reshape = std::make_shared<v1::Reshape>(gather2, row_shape, false);
        auto gather2_reshape2 = std::make_shared<v1::Reshape>(gather2_reshape, col_shape, false);
        auto image_token_mask = make_named_param(element::boolean, Shape{1, 1, 1, SEQ}, "image_token_mask" + suffix);
        auto gather2_select = std::make_shared<v1::Select>(image_token_mask, gather2_reshape2, zeros);
        auto gather2_equal = std::make_shared<v1::Equal>(gather2_select, group_ids);

        auto subg2_bw_and = std::make_shared<v13::BitwiseAnd>(gather1_equal, gather2_equal);
        auto vision_block = std::make_shared<v13::BitwiseAnd>(subg2_bw_and, branch_equal);
        vision_block->set_friendly_name("vision_block" + suffix);

        auto causal_mask = make_named_param(element::boolean, Shape{1, 1, SEQ, SEQ}, "causal_mask" + suffix);
        auto causal_or_vision = std::make_shared<v13::BitwiseOr>(causal_mask, vision_block);
        causal_or_vision->set_friendly_name("causal_or_vision" + suffix);

        params.insert(params.end(), {select_input, image_group_ids, image_token_mask, causal_mask});
        results.push_back(std::make_shared<v0::Result>(causal_or_vision));
    }
}

// Appends the subgraph matched by RemoveTTIShapeOfSubgraph. The `Gather` indices deliberately come
// from an `Add`: that is the node a TTI-free model reshapes against, and therefore the node the pass
// is expected to re-point the ShapeOf at.
void add_shapeof_chain(const std::shared_ptr<v0::Parameter>& tti, ParameterVector& params, ResultVector& results) {
    auto tti_shape_of = std::make_shared<v3::ShapeOf>(tti, element::i64);
    auto tti_gather = std::make_shared<v8::Gather>(tti_shape_of,
                                                   v0::Constant::create(element::i64, Shape{}, {1}),
                                                   v0::Constant::create(element::i32, Shape{}, {0}));
    auto tti_range = make_named_param(element::i64, Shape{1, SEQ}, "tti_range");
    auto tti_less = std::make_shared<v1::Less>(tti_range, tti_gather);
    auto tti_pos_data = make_named_param(element::i64, Shape{1, SEQ}, "tti_pos_data");
    auto tti_select =
        std::make_shared<v1::Select>(tti_less, tti_pos_data, v0::Constant::create(element::i64, Shape{}, {0}));
    auto tti_convert = std::make_shared<v0::Convert>(tti_select, element::i32);
    auto tti_add = std::make_shared<v1::Add>(tti_convert, v0::Constant::create(element::i32, Shape{1, 1}, {1}));
    tti_add->set_friendly_name("tti_add");
    auto tti_shape_of_2 = std::make_shared<v3::ShapeOf>(tti_add, element::i32);
    tti_shape_of_2->set_friendly_name("tti_shape_of_2");

    const auto row_shape = v0::Constant::create(element::i64, Shape{2}, {int64_t{1}, SEQ_I64});

    auto attention_mask = make_named_param(element::i64, Shape{1, SEQ}, "attention_mask");
    auto attn_convert = std::make_shared<v0::Convert>(attention_mask, element::boolean);
    auto attn_reshape = std::make_shared<v1::Reshape>(attn_convert, row_shape, false);

    auto attn_idx_src = make_named_param(element::i32, Shape{1, SEQ}, "attn_idx_src");
    auto attn_idx_add = std::make_shared<v1::Add>(attn_idx_src, v0::Constant::create(element::i32, Shape{1, 1}, {0}));
    attn_idx_add->set_friendly_name("attn_idx_add");

    auto attn_gather =
        std::make_shared<v8::Gather>(attn_reshape, attn_idx_add, v0::Constant::create(element::i32, Shape{}, {1}));
    auto attn_reshape_2 = std::make_shared<v1::Reshape>(attn_gather, row_shape, false);
    auto attn_tti_reshape = std::make_shared<v1::Reshape>(attn_reshape_2, tti_shape_of_2, false);
    attn_tti_reshape->set_friendly_name("attn_tti_reshape");

    auto preceding_mask = make_named_param(element::boolean, Shape{1, 1, SEQ, SEQ}, "preceding_mask");
    auto final_and = std::make_shared<v13::BitwiseAnd>(preceding_mask, attn_tti_reshape);

    params.insert(params.end(), {tti_range, tti_pos_data, attention_mask, attn_idx_src, preceding_mask});
    results.push_back(std::make_shared<v0::Result>(final_and));
}

std::shared_ptr<ov::Model> make_vision_model() {
    auto tti = make_token_type_ids();
    ParameterVector params{tti};
    ResultVector results;
    add_vision_chain(tti, params, results);
    return std::make_shared<ov::Model>(results, params, "gemma3_tti_vision");
}

std::shared_ptr<ov::Model> make_shapeof_model() {
    auto tti = make_token_type_ids();
    ParameterVector params{tti};
    ResultVector results;
    add_shapeof_chain(tti, params, results);
    return std::make_shared<ov::Model>(results, params, "gemma3_tti_shapeof");
}

std::shared_ptr<ov::Model> make_combined_model() {
    auto tti = make_token_type_ids();
    ParameterVector params{tti};
    ResultVector results;
    add_vision_chain(tti, params, results);
    add_shapeof_chain(tti, params, results);
    return std::make_shared<ov::Model>(results, params, "gemma3_tti_combined");
}

// Both known patterns plus a `token_type_ids` reader neither of them covers - stands in for e.g. an
// attention branch whose vision subgraph deviates from the matched topology.
std::shared_ptr<ov::Model> make_model_with_unmatched_consumer() {
    auto tti = make_token_type_ids();
    ParameterVector params{tti};
    ResultVector results;
    add_vision_chain(tti, params, results);
    add_shapeof_chain(tti, params, results);

    auto unmatched = std::make_shared<v1::Add>(tti, v0::Constant::create(element::i64, Shape{1, 1}, {1}));
    unmatched->set_friendly_name("unmatched_tti_consumer");
    results.push_back(std::make_shared<v0::Result>(unmatched));
    return std::make_shared<ov::Model>(results, params, "gemma3_tti_unmatched_consumer");
}

bool has_parameter_with_name(const std::shared_ptr<ov::Model>& model, const std::string& name) {
    const auto& params = model->get_parameters();
    return std::any_of(params.begin(), params.end(), [&name](const auto& param) {
        return param->get_friendly_name() == name;
    });
}

std::shared_ptr<ov::Node> find_node(const std::shared_ptr<ov::Model>& model, const std::string& name) {
    for (const auto& op : model->get_ordered_ops()) {
        if (op->get_friendly_name() == name) {
            return op;
        }
    }
    return nullptr;
}

template <typename T>
size_t count_ops_of_type(const std::shared_ptr<ov::Model>& model) {
    const auto ops = model->get_ordered_ops();
    return static_cast<size_t>(std::count_if(ops.begin(), ops.end(), [](const auto& op) {
        return ov::is_type<T>(op);
    }));
}

// ---------------------------------------------------------------------------
// Preconditions on the test models themselves
// ---------------------------------------------------------------------------

TEST(RemoveTokenTypeIdsTest, TestModelsAreValid) {
    for (const auto& model : {make_vision_model(), make_shapeof_model(), make_combined_model()}) {
        EXPECT_NO_THROW(model->validate_nodes_and_infer_types()) << "model " << model->get_friendly_name();
        EXPECT_TRUE(has_parameter_with_name(model, "token_type_ids")) << "model " << model->get_friendly_name();
    }
}

// ---------------------------------------------------------------------------
// Negative case: nothing to do
// ---------------------------------------------------------------------------

TEST(RemoveTokenTypeIdsTest, NoOpWhenTokenTypeIdsAbsent) {
    auto input = make_named_param(element::f32, Shape{1, SEQ}, "input_ids");
    auto model = std::make_shared<ov::Model>(ResultVector{std::make_shared<v0::Result>(input)},
                                             ParameterVector{input},
                                             "no_tti_model");

    ov::npuw::RemoveTokenTypeIds pass;
    EXPECT_FALSE(pass.run_on_model(model));
    EXPECT_TRUE(has_parameter_with_name(model, "input_ids"));
}

// ---------------------------------------------------------------------------
// RemoveTTIVisionSubgraph
// ---------------------------------------------------------------------------

TEST(RemoveTokenTypeIdsTest, VisionPattern_VisionBlockReplacedWithFalseConstant) {
    auto model = make_vision_model();

    ov::npuw::RemoveTokenTypeIds pass;
    EXPECT_TRUE(pass.run_on_model(model));

    size_t bw_or_count = 0;
    for (const auto& op : model->get_ordered_ops()) {
        if (!ov::is_type<v13::BitwiseOr>(op)) {
            continue;
        }
        ++bw_or_count;
        auto vision_input = ov::as_type_ptr<v0::Constant>(op->input_value(1).get_node_shared_ptr());
        ASSERT_NE(vision_input, nullptr) << "BitwiseOr input[1] must be replaced with a Constant";
        EXPECT_EQ(vision_input->get_element_type(), element::boolean);
        const auto values = vision_input->cast_vector<bool>();
        ASSERT_EQ(values.size(), 1u);
        EXPECT_FALSE(values[0]) << "the vision block must be neutralized with `false`";
    }
    EXPECT_EQ(bw_or_count, 2u) << "both the global and the local attention branch must be patched";
    EXPECT_NO_THROW(model->validate_nodes_and_infer_types());
}

TEST(RemoveTokenTypeIdsTest, VisionPattern_CausalMaskInputPreserved) {
    auto model = make_vision_model();

    ov::npuw::RemoveTokenTypeIds pass;
    ASSERT_TRUE(pass.run_on_model(model));

    std::set<std::string> causal_inputs;
    for (const auto& op : model->get_ordered_ops()) {
        if (ov::is_type<v13::BitwiseOr>(op)) {
            causal_inputs.insert(op->input_value(0).get_node()->get_friendly_name());
        }
    }
    const std::set<std::string> expected{"causal_mask_global", "causal_mask_local"};
    EXPECT_EQ(causal_inputs, expected) << "input[0] of each BitwiseOr must still be the causal mask";
}

TEST(RemoveTokenTypeIdsTest, VisionPattern_UnusedSubgraphIsRemoved) {
    auto model = make_vision_model();
    ASSERT_GT(count_ops_of_type<v0::CumSum>(model), 0u);
    ASSERT_GT(count_ops_of_type<v12::Pad>(model), 0u);

    ov::npuw::RemoveTokenTypeIds pass;
    ASSERT_TRUE(pass.run_on_model(model));

    // Neutralizing the merge points drops the last consumer of the whole vision subgraph.
    EXPECT_EQ(count_ops_of_type<v0::CumSum>(model), 0u);
    EXPECT_EQ(count_ops_of_type<v12::Pad>(model), 0u);
    EXPECT_EQ(count_ops_of_type<v13::BitwiseNot>(model), 0u);
    EXPECT_FALSE(has_parameter_with_name(model, "token_type_ids"));
}

// ---------------------------------------------------------------------------
// RemoveTTIShapeOfSubgraph
// ---------------------------------------------------------------------------

TEST(RemoveTokenTypeIdsTest, ShapeOfPattern_ShapeOfRedirectedToGatherIndices) {
    auto model = make_shapeof_model();

    auto shape_of_2 = find_node(model, "tti_shape_of_2");
    ASSERT_NE(shape_of_2, nullptr);
    ASSERT_EQ(shape_of_2->input_value(0).get_node()->get_friendly_name(), "tti_add")
        << "before the pass the Reshape target shape is derived from `token_type_ids`";

    ov::npuw::RemoveTokenTypeIds pass;
    EXPECT_TRUE(pass.run_on_model(model));

    shape_of_2 = find_node(model, "tti_shape_of_2");
    ASSERT_NE(shape_of_2, nullptr) << "the ShapeOf itself must survive - only its input is re-pointed";
    EXPECT_EQ(shape_of_2->input_value(0).get_node()->get_friendly_name(), "attn_idx_add")
        << "after the pass it must be derived from the attention_mask Gather indices";

    EXPECT_FALSE(has_parameter_with_name(model, "token_type_ids"));
    EXPECT_NO_THROW(model->validate_nodes_and_infer_types());
}

TEST(RemoveTokenTypeIdsTest, ShapeOfPattern_AttentionMaskChainPreserved) {
    auto model = make_shapeof_model();
    const auto reshapes_before = count_ops_of_type<v1::Reshape>(model);
    const auto gathers_before = count_ops_of_type<v8::Gather>(model);

    ov::npuw::RemoveTokenTypeIds pass;
    ASSERT_TRUE(pass.run_on_model(model));

    // Only the TTI-side Gather goes away; every Reshape of the attention_mask chain stays.
    EXPECT_EQ(count_ops_of_type<v1::Reshape>(model), reshapes_before);
    EXPECT_EQ(count_ops_of_type<v8::Gather>(model), gathers_before - 1);
    EXPECT_NE(find_node(model, "attn_tti_reshape"), nullptr);
    EXPECT_NE(find_node(model, "attn_idx_add"), nullptr);
    EXPECT_TRUE(has_parameter_with_name(model, "attention_mask"));
    EXPECT_TRUE(has_parameter_with_name(model, "attn_idx_src"));
    EXPECT_TRUE(has_parameter_with_name(model, "preceding_mask"));
}

// ---------------------------------------------------------------------------
// Both patterns together
// ---------------------------------------------------------------------------

TEST(RemoveTokenTypeIdsTest, CombinedPatterns_TokenTypeIdsRemoved) {
    auto model = make_combined_model();
    const auto params_before = model->get_parameters().size();

    ov::npuw::RemoveTokenTypeIds pass;
    EXPECT_TRUE(pass.run_on_model(model));

    EXPECT_FALSE(has_parameter_with_name(model, "token_type_ids"));
    EXPECT_EQ(model->get_parameters().size(), params_before - 1)
        << "`token_type_ids` must be the only parameter removed";
    for (const char* name : {"range_input",
                             "select_input_global",
                             "image_group_ids_global",
                             "image_token_mask_global",
                             "causal_mask_global",
                             "select_input_local",
                             "image_group_ids_local",
                             "image_token_mask_local",
                             "causal_mask_local",
                             "tti_range",
                             "tti_pos_data",
                             "attention_mask",
                             "attn_idx_src",
                             "preceding_mask"}) {
        EXPECT_TRUE(has_parameter_with_name(model, name)) << "parameter `" << name << "` must be preserved";
    }
    EXPECT_NO_THROW(model->validate_nodes_and_infer_types());
}

// ---------------------------------------------------------------------------
// Removal contract: a consumer surviving the rewrites is a hard error
// ---------------------------------------------------------------------------

TEST(RemoveTokenTypeIdsTest, ThrowsWhenUnmatchedConsumerRemains) {
    auto model = make_model_with_unmatched_consumer();

    // Both known patterns fire and rewrite the graph, but an unmatched reader still holds
    // `token_type_ids`. The model is partially transformed by now - one attention branch would keep
    // its vision mask while another lost it - so the pass must fail loudly rather than hand back a
    // model that infers and silently produces wrong results.
    ov::npuw::RemoveTokenTypeIds pass;
    try {
        pass.run_on_model(model);
        FAIL() << "expected a throw on a `token_type_ids` consumer that neither pattern covers";
    } catch (const ov::Exception& e) {
        // The diagnostic has to name the offender - that is the only lead for extending the pattern.
        const std::string message = e.what();
        EXPECT_NE(message.find("token_type_ids"), std::string::npos) << message;
        EXPECT_NE(message.find("unmatched_tti_consumer"), std::string::npos) << message;
    }
}

TEST(RemoveTokenTypeIdsTest, SecondRunIsNoOp) {
    auto model = make_combined_model();

    ov::npuw::RemoveTokenTypeIds first;
    ASSERT_TRUE(first.run_on_model(model));
    ASSERT_FALSE(has_parameter_with_name(model, "token_type_ids"));

    ov::npuw::RemoveTokenTypeIds second;
    EXPECT_FALSE(second.run_on_model(model));
    EXPECT_NO_THROW(model->validate_nodes_and_infer_types());
}

}  // namespace
