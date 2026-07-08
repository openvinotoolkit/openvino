// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "npuw_transformations/remove_token_type_ids.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/opsets/opset13.hpp"

// Tests for ov::npuw::RemoveTokenTypeIds pass.
//
// The model built here mirrors the Gemma-3 token_type_ids blockwise attention
// subgraph with two branches (local and global attention).
//
// Subgraph 1 (blockwise vision mask, shared up to Less):
//   token_type_ids -> Equal -> Pad -> Slice -> BitwiseNot
//                     |                          |
//                     +-----> BitwiseAnd <-------+
//                                 |
//                              Convert -> CumSum -> Add -> Convert
//                                                            |
//   Equal --------> Select(Equal, Convert, broadcast_val) <--+
//                      |
//                   ShapeOf -> Gather
//                                |
//                              Less -+-> Select_A -> Equal_A -> BitwiseAnd_A (vision_block_A)
//                                    |
//                                    +-> Select_B -> Equal_B -> BitwiseAnd_B (vision_block_B)
//
// Subgraph 2 (index reshape from token_type_ids shape):
//   token_type_ids -> ShapeOf -> Gather -> Less -> Select -> Convert -> Add -> ShapeOf
//                                                                                |
//                                                                     +-> Reshape_A
//                                                                     +-> Reshape_B
//
// Merge (per branch):
//   BitwiseOr(causal_mask, vision_block) -> BitwiseAnd(true_mask, BitwiseOr) -> BitwiseAnd(prev, reshape)
//
// After the pass:
//   - Both BitwiseOr input[1] (vision_block) nodes are replaced with Constant(false)
//   - Both final BitwiseAnd input[1] (reshape) nodes are replaced with Constant(true)
//   - token_type_ids parameter is removed from the model

namespace {

using namespace ov;
using namespace ov::op;
using namespace ov::opset13;

// Build a model that replicates the Gemma-3 token_type_ids blockwise attention subgraph
// with two branches (local and global attention), matching the real model topology.
// After `Less`, there are two independent paths:
//   Branch A (global attention): Less -> Select_A -> Equal_A -> BitwiseAnd_A (vision_block_A)
//   Branch B (local attention):  Less -> Select_B -> Equal_B -> BitwiseAnd_B (vision_block_B)
// Each branch has its own merge point:
//   BitwiseOr(causal_mask, vision_block) -> BitwiseAnd(true_mask, BitwiseOr) -> BitwiseAnd(prev, reshape)
// Subgraph 2's ShapeOf also feeds two Reshape nodes (one per branch).
static std::shared_ptr<ov::Model> make_gemma3_tti_subgraph_model() {
    // token_type_ids: [batch, seq_len]
    auto token_type_ids = std::make_shared<v0::Parameter>(element::i64, Shape{1, 128});
    token_type_ids->set_friendly_name("token_type_ids");
    token_type_ids->output(0).set_names({"token_type_ids"});

    // ======== Subgraph 1: blockwise vision mask (shared part) ========

    // Equal(token_type_ids, constant) -> bool
    auto equal_const = v0::Constant::create(element::i64, Shape{1, 1}, {2});
    auto subg1_equal = std::make_shared<v1::Equal>(token_type_ids, equal_const);

    // Pad(equal, pads_begin, pads_end, pad_value)
    auto pads_begin = v0::Constant::create(element::i64, Shape{2}, {0, 1});
    auto pads_end = v0::Constant::create(element::i64, Shape{2}, {0, 0});
    auto pad_value = v0::Constant::create(element::boolean, Shape{}, {false});
    auto subg1_pad = std::make_shared<v12::Pad>(subg1_equal, pads_begin, pads_end, pad_value,
                                                 ov::op::PadMode::CONSTANT);

    // Slice(pad, start, stop, step, axes)
    auto slice_start = v0::Constant::create(element::i64, Shape{1}, {1});
    auto slice_stop = v0::Constant::create(element::i64, Shape{1}, {129});
    auto slice_step = v0::Constant::create(element::i64, Shape{1}, {1});
    auto slice_axes = v0::Constant::create(element::i64, Shape{1}, {1});
    auto subg1_slice = std::make_shared<v8::Slice>(subg1_pad, slice_start, slice_stop, slice_step, slice_axes);

    // BitwiseNot(slice)
    auto subg1_bw_not = std::make_shared<v13::BitwiseNot>(subg1_slice);

    // BitwiseAnd(equal, bw_not)
    auto subg1_bw_and = std::make_shared<v13::BitwiseAnd>(subg1_equal, subg1_bw_not);

    // Convert(bw_and) -> i32
    auto subg1_convert = std::make_shared<v0::Convert>(subg1_bw_and, element::i32);

    // CumSum(convert, axis)
    auto cumsum_axis = v0::Constant::create(element::i64, Shape{}, {1});
    auto subg1_cumsum = std::make_shared<v0::CumSum>(subg1_convert, cumsum_axis);

    // Add(cumsum, const) - subtract 1 in the original model (Add with -1)
    auto add_const = v0::Constant::create(element::i32, Shape{1, 1}, {-1});
    auto subg1_add = std::make_shared<v1::Add>(subg1_cumsum, add_const);

    // Convert(add) -> i64
    auto subg1_convert_add = std::make_shared<v0::Convert>(subg1_add, element::i64);

    // Select(equal, convert_add, broadcast_val)
    auto broadcast_val = v0::Constant::create(element::i64, Shape{1, 1}, {0});
    auto subg1_select = std::make_shared<v1::Select>(subg1_equal, subg1_convert_add, broadcast_val);

    // ShapeOf(select)
    auto subg1_shape_of = std::make_shared<v3::ShapeOf>(subg1_select, element::i64);

    // Gather(shape_of, indices, axis)
    auto gather_indices = v0::Constant::create(element::i64, Shape{}, {1});
    auto gather_axis = v0::Constant::create(element::i32, Shape{}, {0});
    auto subg1_gather = std::make_shared<v8::Gather>(subg1_shape_of, gather_indices, gather_axis);

    // Less(range, gather) - shared between both branches
    auto range_input = std::make_shared<v0::Parameter>(element::i64, Shape{1, 1, 1, 128});
    range_input->set_friendly_name("range_input");
    range_input->output(0).set_names({"range_input"});
    auto subg1_less = std::make_shared<v1::Less>(range_input, subg1_gather);

    // ======== Branch A (global attention) ========

    auto select_input_a = std::make_shared<v0::Parameter>(element::i64, Shape{1, 1, 1, 128});
    select_input_a->set_friendly_name("select_input_a");
    select_input_a->output(0).set_names({"select_input_a"});
    auto select_zeros_a = v0::Constant::create(element::i64, Shape{}, {0});
    auto branch_a_select = std::make_shared<v1::Select>(subg1_less, select_input_a, select_zeros_a);

    auto image_group_ids_a = std::make_shared<v0::Parameter>(element::i64, Shape{1, 1, 128, 1});
    image_group_ids_a->set_friendly_name("image_group_ids_a");
    image_group_ids_a->output(0).set_names({"image_group_ids_a"});
    auto branch_a_equal = std::make_shared<v1::Equal>(image_group_ids_a, branch_a_select);

    auto is_image_block_a = std::make_shared<v0::Parameter>(element::boolean, Shape{1, 1, 128, 128});
    is_image_block_a->set_friendly_name("is_image_block_a");
    is_image_block_a->output(0).set_names({"is_image_block_a"});
    auto vision_block_a = std::make_shared<v13::BitwiseAnd>(is_image_block_a, branch_a_equal);

    // ======== Branch B (local/sliding window attention) ========

    auto select_input_b = std::make_shared<v0::Parameter>(element::i64, Shape{1, 1, 1, 128});
    select_input_b->set_friendly_name("select_input_b");
    select_input_b->output(0).set_names({"select_input_b"});
    auto select_zeros_b = v0::Constant::create(element::i64, Shape{}, {0});
    auto branch_b_select = std::make_shared<v1::Select>(subg1_less, select_input_b, select_zeros_b);

    auto image_group_ids_b = std::make_shared<v0::Parameter>(element::i64, Shape{1, 1, 128, 1});
    image_group_ids_b->set_friendly_name("image_group_ids_b");
    image_group_ids_b->output(0).set_names({"image_group_ids_b"});
    auto branch_b_equal = std::make_shared<v1::Equal>(image_group_ids_b, branch_b_select);

    auto is_image_block_b = std::make_shared<v0::Parameter>(element::boolean, Shape{1, 1, 128, 128});
    is_image_block_b->set_friendly_name("is_image_block_b");
    is_image_block_b->output(0).set_names({"is_image_block_b"});
    auto vision_block_b = std::make_shared<v13::BitwiseAnd>(is_image_block_b, branch_b_equal);

    // ======== Subgraph 2: index reshape from token_type_ids shape (shared part) ========

    // ShapeOf(token_type_ids)
    auto subg2_shape_of = std::make_shared<v3::ShapeOf>(token_type_ids, element::i64);

    // Gather(shape_of, index, axis)
    auto subg2_gather_idx = v0::Constant::create(element::i64, Shape{}, {1});
    auto subg2_gather_axis = v0::Constant::create(element::i32, Shape{}, {0});
    auto subg2_gather = std::make_shared<v8::Gather>(subg2_shape_of, subg2_gather_idx, subg2_gather_axis);

    // Less(range, gather)
    auto subg2_range = std::make_shared<v0::Parameter>(element::i64, Shape{1, 128});
    subg2_range->set_friendly_name("subg2_range");
    subg2_range->output(0).set_names({"subg2_range"});
    auto subg2_less = std::make_shared<v1::Less>(subg2_range, subg2_gather);

    // Select(less, position_data, zeros)
    auto subg2_pos_data = std::make_shared<v0::Parameter>(element::i64, Shape{1, 128});
    subg2_pos_data->set_friendly_name("subg2_pos_data");
    subg2_pos_data->output(0).set_names({"subg2_pos_data"});
    auto subg2_zeros = v0::Constant::create(element::i64, Shape{}, {0});
    auto subg2_select = std::make_shared<v1::Select>(subg2_less, subg2_pos_data, subg2_zeros);

    // Convert(select) -> i32
    auto subg2_convert = std::make_shared<v0::Convert>(subg2_select, element::i32);

    // Add(convert, const)
    auto subg2_add_const = v0::Constant::create(element::i32, Shape{1, 1}, {1});
    auto subg2_add = std::make_shared<v1::Add>(subg2_convert, subg2_add_const);

    // ShapeOf(add) -> shared, feeds two Reshape nodes (one per branch)
    auto subg2_shape_of_2 = std::make_shared<v3::ShapeOf>(subg2_add, element::i32);

    // Branch A reshape
    auto reshape_data_a = std::make_shared<v0::Parameter>(element::boolean, Shape{128});
    reshape_data_a->set_friendly_name("reshape_data_a");
    reshape_data_a->output(0).set_names({"reshape_data_a"});
    auto subg2_reshape_a = std::make_shared<v1::Reshape>(reshape_data_a, subg2_shape_of_2, false);

    // Branch B reshape
    auto reshape_data_b = std::make_shared<v0::Parameter>(element::boolean, Shape{128});
    reshape_data_b->set_friendly_name("reshape_data_b");
    reshape_data_b->output(0).set_names({"reshape_data_b"});
    auto subg2_reshape_b = std::make_shared<v1::Reshape>(reshape_data_b, subg2_shape_of_2, false);

    // ======== Merge point A (global attention) ========

    auto causal_mask_a = std::make_shared<v0::Parameter>(element::boolean, Shape{1, 1, 128, 128});
    causal_mask_a->set_friendly_name("causal_mask_a");
    causal_mask_a->output(0).set_names({"causal_mask_a"});
    auto bw_or_a = std::make_shared<v13::BitwiseOr>(causal_mask_a, vision_block_a);

    auto true_mask_a = std::make_shared<v0::Parameter>(element::boolean, Shape{});
    true_mask_a->set_friendly_name("true_mask_a");
    true_mask_a->output(0).set_names({"true_mask_a"});
    auto bw_and_a = std::make_shared<v13::BitwiseAnd>(true_mask_a, bw_or_a);

    auto final_bw_and_a = std::make_shared<v13::BitwiseAnd>(bw_and_a, subg2_reshape_a);

    // ======== Merge point B (local/sliding attention) ========

    auto causal_mask_b = std::make_shared<v0::Parameter>(element::boolean, Shape{1, 1, 128, 128});
    causal_mask_b->set_friendly_name("causal_mask_b");
    causal_mask_b->output(0).set_names({"causal_mask_b"});
    auto bw_or_b = std::make_shared<v13::BitwiseOr>(causal_mask_b, vision_block_b);

    auto true_mask_b = std::make_shared<v0::Parameter>(element::boolean, Shape{});
    true_mask_b->set_friendly_name("true_mask_b");
    true_mask_b->output(0).set_names({"true_mask_b"});
    auto bw_and_b = std::make_shared<v13::BitwiseAnd>(true_mask_b, bw_or_b);

    auto final_bw_and_b = std::make_shared<v13::BitwiseAnd>(bw_and_b, subg2_reshape_b);

    // ======== Results ========

    auto result_a = std::make_shared<v0::Result>(final_bw_and_a);
    auto result_b = std::make_shared<v0::Result>(final_bw_and_b);

    return std::make_shared<ov::Model>(
        ov::ResultVector{result_a, result_b},
        ov::ParameterVector{token_type_ids, range_input,
                            select_input_a, image_group_ids_a, is_image_block_a,
                            select_input_b, image_group_ids_b, is_image_block_b,
                            subg2_range, subg2_pos_data, reshape_data_a, reshape_data_b,
                            causal_mask_a, true_mask_a, causal_mask_b, true_mask_b},
        "gemma3_tti_subgraph_test");
}

// Count nodes of a specific op type in the model.
template <typename T>
static size_t count_ops_of_type(const std::shared_ptr<ov::Model>& model) {
    size_t count = 0;
    for (const auto& op : model->get_ordered_ops()) {
        if (ov::is_type<T>(op))
            ++count;
    }
    return count;
}

static bool has_parameter_with_name(const std::shared_ptr<ov::Model>& model, const std::string& name) {
    for (const auto& param : model->get_parameters()) {
        if (param->get_friendly_name() == name) {
            return true;
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// Test 1: The pass fires and removes token_type_ids parameter.
// ---------------------------------------------------------------------------
TEST(RemoveTokenTypeIdsTest, PassFiresAndRemovesParameter) {
    auto model = make_gemma3_tti_subgraph_model();

    // Precondition: token_type_ids parameter exists
    ASSERT_TRUE(has_parameter_with_name(model, "token_type_ids"));

    ov::npuw::RemoveTokenTypeIds pass;
    EXPECT_TRUE(pass.run_on_model(model));

    // token_type_ids parameter must be removed
    EXPECT_FALSE(has_parameter_with_name(model, "token_type_ids"));
}

// ---------------------------------------------------------------------------
// Test 2: After the pass, both BitwiseOr nodes' second input (vision_block)
//         is replaced with a Constant(false).
// ---------------------------------------------------------------------------
TEST(RemoveTokenTypeIdsTest, VisionBlockReplacedWithFalse) {
    auto model = make_gemma3_tti_subgraph_model();

    ov::npuw::RemoveTokenTypeIds pass;
    pass.run_on_model(model);

    // Find all BitwiseOr nodes in the transformed model — both branches must be patched
    size_t bw_or_count = 0;
    for (const auto& op : model->get_ordered_ops()) {
        if (!ov::is_type<v13::BitwiseOr>(op))
            continue;
        ++bw_or_count;
        // Input 1 should now be a Constant with value false
        auto input1 = op->input_value(1).get_node_shared_ptr();
        EXPECT_TRUE(ov::is_type<v0::Constant>(input1))
            << "BitwiseOr input[1] must be replaced with a Constant";
        auto c = ov::as_type_ptr<v0::Constant>(input1);
        ASSERT_NE(c, nullptr);
        EXPECT_EQ(c->get_element_type(), element::boolean);
        auto val = c->cast_vector<bool>();
        ASSERT_EQ(val.size(), 1u);
        EXPECT_FALSE(val[0]) << "Vision block replacement must be false";
    }
    EXPECT_EQ(bw_or_count, 2u) << "Both BitwiseOr nodes (global + local) must be patched";
}

// ---------------------------------------------------------------------------
// Test 3: After the pass, both final BitwiseAnd nodes' second input (reshape
//         from subg2) is replaced with a Constant(true).
// ---------------------------------------------------------------------------
TEST(RemoveTokenTypeIdsTest, FinalBitwiseAndInputReplacedWithTrue) {
    auto model = make_gemma3_tti_subgraph_model();

    ov::npuw::RemoveTokenTypeIds pass;
    pass.run_on_model(model);

    // Both result nodes' input chains should end with BitwiseAnd whose input[1] is Constant(true)
    auto results = model->get_results();
    ASSERT_EQ(results.size(), 2u);

    for (size_t i = 0; i < results.size(); ++i) {
        auto result_input = results[i]->input_value(0).get_node_shared_ptr();
        ASSERT_TRUE(ov::is_type<v13::BitwiseAnd>(result_input))
            << "Result[" << i << "]'s input must still be BitwiseAnd";

        auto final_and_input1 = result_input->input_value(1).get_node_shared_ptr();
        EXPECT_TRUE(ov::is_type<v0::Constant>(final_and_input1))
            << "Final BitwiseAnd input[1] for result[" << i << "] must be replaced with a Constant";
        auto c = ov::as_type_ptr<v0::Constant>(final_and_input1);
        ASSERT_NE(c, nullptr);
        EXPECT_EQ(c->get_element_type(), element::boolean);
        auto val = c->cast_vector<bool>();
        ASSERT_EQ(val.size(), 1u);
        EXPECT_TRUE(val[0]) << "Final BitwiseAnd replacement for result[" << i << "] must be true";
    }
}

// ---------------------------------------------------------------------------
// Test 4: When token_type_ids is absent, the pass returns false (no-op).
// ---------------------------------------------------------------------------
TEST(RemoveTokenTypeIdsTest, NoOpWhenTokenTypeIdsAbsent) {
    // Build a trivial model without token_type_ids
    auto input = std::make_shared<v0::Parameter>(element::f32, Shape{1, 128});
    input->set_friendly_name("input_ids");
    input->output(0).set_names({"input_ids"});
    auto result = std::make_shared<v0::Result>(input);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input}, "no_tti_model");

    ov::npuw::RemoveTokenTypeIds pass;
    EXPECT_FALSE(pass.run_on_model(model));
}

// ---------------------------------------------------------------------------
// Test 5: After the pass, all other parameters (except token_type_ids) remain.
// ---------------------------------------------------------------------------
TEST(RemoveTokenTypeIdsTest, OtherParametersPreserved) {
    auto model = make_gemma3_tti_subgraph_model();
    const auto initial_param_count = model->get_parameters().size();

    ov::npuw::RemoveTokenTypeIds pass;
    pass.run_on_model(model);

    // Exactly one parameter (token_type_ids) should have been removed
    EXPECT_EQ(model->get_parameters().size(), initial_param_count - 1);

    // Verify key parameters still exist
    EXPECT_TRUE(has_parameter_with_name(model, "range_input"));
    EXPECT_TRUE(has_parameter_with_name(model, "causal_mask"));
    EXPECT_TRUE(has_parameter_with_name(model, "true_mask"));
}

}  // namespace
