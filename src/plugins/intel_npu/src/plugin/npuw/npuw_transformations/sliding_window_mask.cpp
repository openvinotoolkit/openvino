// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sliding_window_mask.hpp"

#include "../logging.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace opp = ov::pass::pattern;

namespace {

// diagnostics warnings on OPENVINO_MATCHER_PASS_RTTI() definition: visibility hidden
#ifdef __GNUC__
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wattributes"
#endif

// ============================================================================
// Shared helper: Rebuild sliding window attention mask for static KV buffers
//                with different paddings.
// ============================================================================
// This function implements the common 4-step transformation used by
// Phi3, Gemma2, Gemma3 or Gemma4 sliding window attention patterns:
//
// 1. (K > pos_ids - window) & (K <= Q_range)
//    -- Use temporal position_ids for the left window bound to handle right-padded
//       past tokens, while preserving the causal mask for present tokens.
// 2. (K > Q_range - window) | (K < past_kv_len)
//    -- Bound present tokens by window size while always allowing past tokens.
// 3. Result = 1 & 2
//    -- Together they form the correct sliding window mask for past and present
//       tokens and the causal mask for present tokens.
// 4. Clean = 3 & attention_mask[past_kv_len:].T
//    -- Remove padding artifacts.
void rebuild_sliding_window_mask(const std::shared_ptr<ov::Node>& attention_mask_node_ptr,
                                 const std::shared_ptr<ov::Node>& position_ids_node_ptr,
                                 const std::shared_ptr<ov::Node>& matched_past_kv_len,
                                 const std::shared_ptr<ov::Node>& matched_full_ctx_len,
                                 const std::shared_ptr<ov::Node>& matched_key_range_row,
                                 const std::shared_ptr<ov::Node>& matched_neg_window_size,
                                 const std::shared_ptr<ov::Node>& matched_sliding_mask,
                                 const std::shared_ptr<ov::Node>& matched_sliding_and_causal_mask,
                                 const char* log_prefix) {
    auto neg_window_size_const = std::static_pointer_cast<ov::op::v0::Constant>(matched_neg_window_size);
    OPENVINO_ASSERT(neg_window_size_const->get_output_size() == 1,
                    "Sliding window size constant must be of size 1, but got " +
                        std::to_string(neg_window_size_const->get_output_size()));

    OPENVINO_ASSERT(attention_mask_node_ptr, "Passed attention_mask node is nullptr!");
    OPENVINO_ASSERT(position_ids_node_ptr, "Passed position_ids node is nullptr!");

    // Create constants
    auto const_zero = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, 0);

    // =======================================================================
    // STEP 1: (K > pos_ids - window) & (K <= Q_range)
    //         Use temporal position_ids for the left window bound to correctly handle
    //         right-padded past tokens, while preserving the causal mask for present tokens.
    // =======================================================================
    std::shared_ptr<ov::Node> query_range_as_pos_ids = position_ids_node_ptr;
    if (neg_window_size_const->output(0).get_element_type() == ov::element::f32) {
        query_range_as_pos_ids = std::make_shared<ov::op::v0::Convert>(position_ids_node_ptr, ov::element::f32);
    }
    auto query_range_as_pos_ids_unsqueezed =
        std::make_shared<ov::op::v0::Unsqueeze>(query_range_as_pos_ids, const_zero);
    auto const_three = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, 3);
    auto query_range_as_pos_ids_col =
        std::make_shared<ov::op::v0::Unsqueeze>(query_range_as_pos_ids_unsqueezed, const_three);
    auto query_range_as_pos_left_bound =
        std::make_shared<ov::op::v1::Add>(query_range_as_pos_ids_col, matched_neg_window_size);
    auto sliding_mask_for_right_padding =
        std::make_shared<ov::op::v1::Greater>(matched_key_range_row, query_range_as_pos_left_bound);
    matched_sliding_and_causal_mask->input(0).replace_source_output(sliding_mask_for_right_padding);

    // =======================================================================
    // STEP 2: (K > Q_range - window) | (K < past_kv_len)
    //         Bound present tokens by window size while always allowing
    //         past prefill tokens.
    // =======================================================================
    std::shared_ptr<ov::Node> past_kv_len_argument = matched_past_kv_len;
    if (neg_window_size_const->output(0).get_element_type() == ov::element::f32) {
        past_kv_len_argument = std::make_shared<ov::op::v0::Convert>(matched_past_kv_len, ov::element::f32);
    }
    auto only_past_tokens_mask = std::make_shared<ov::op::v1::Less>(matched_key_range_row, past_kv_len_argument);
    auto sliding_mask_for_left_padding_or_only_past =
        std::make_shared<ov::op::v13::BitwiseOr>(matched_sliding_mask, only_past_tokens_mask);

    // =======================================================================
    // STEP 3: Result = 1 & 2
    // =======================================================================
    // NB: target_inputs must be captured BEFORE new_sliding_and_causal_mask is
    // created, because creating it registers it as a consumer of
    // matched_sliding_and_causal_mask. Capturing after would include the new
    // node in target_inputs and cause a graph cycle when replacing outputs.
    auto target_inputs = matched_sliding_and_causal_mask->output(0).get_target_inputs();
    auto new_sliding_and_causal_mask =
        std::make_shared<ov::op::v13::BitwiseAnd>(matched_sliding_and_causal_mask,
                                                  sliding_mask_for_left_padding_or_only_past);

    // =======================================================================
    // STEP 4: Clean = 3 & attention_mask[past_kv_len:].T
    //         Remove padding artifacts.
    // =======================================================================
    std::vector<int64_t> shape_rank_one{1};
    auto shape_rank_one_const = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, shape_rank_one);
    auto past_len_reshaped = std::make_shared<ov::op::v1::Reshape>(matched_past_kv_len, shape_rank_one_const, false);
    auto full_ctx_len_reshaped =
        std::make_shared<ov::op::v1::Reshape>(matched_full_ctx_len, shape_rank_one_const, false);
    auto const_one_rank_one = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, 1);
    auto attention_mask_bool = std::make_shared<ov::op::v0::Convert>(attention_mask_node_ptr, ov::element::boolean);
    auto present_atten_mask_bool = std::make_shared<ov::op::v8::Slice>(attention_mask_bool,
                                                                       past_len_reshaped,
                                                                       full_ctx_len_reshaped,
                                                                       const_one_rank_one,
                                                                       const_one_rank_one);
    std::vector<int64_t> vector_shape{-1, 1};
    auto vector_shape_const = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{2}, vector_shape);
    auto present_atten_mask_bool_col =
        std::make_shared<ov::op::v1::Reshape>(present_atten_mask_bool, vector_shape_const, false);
    auto clean_sliding_and_causal_mask =
        std::make_shared<ov::op::v13::BitwiseAnd>(new_sliding_and_causal_mask, present_atten_mask_bool_col);

    // Replace all target inputs
    for (auto&& input : target_inputs) {
        input.replace_source_output(clean_sliding_and_causal_mask);
    }

    LOG_INFO(std::string(log_prefix) + " sliding window attention mask pattern found and patched.");
}

class OldPhi3SlidingMaskMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::npuw::patterns::OldPhi3SlidingMaskMatcher");

    OldPhi3SlidingMaskMatcher() {
        // Search for the Phi3 old sliding mask pattern to extend it to work with right-padded
        // past tokens and left-padded present tokens.
        //
        // Mask creation is simply done via "less_equal" and "greater" operations between
        // row K range: [0,... mask_len] and column Q range: [current_pos_id,... mask_len].T
        // and sliding window length.
        // Due to broadcasting rules these two operation form two triangular masks.
        //
        // -  "less_equal" forms a sliding window mask, more precisely, it has following expression:
        //
        //        row range [0,... mask_len] <= column range [current_pos_id - sliding_window_size,
        //                                                    ...,
        //                                                    mask_len    -    sliding_window_size]
        //
        //       forming, under example conditions, the mask below:
        //        past tokens = 3
        //        present tokens = 5 (starting with current_pos_id = 3)
        //        sliding window len = 4
        //                    K0 K1 K2 K3 K4 K5 K6 K7
        //                   [ 0  1  2  3  4  5  6  7 ]
        //        Q3[ 3 - 4 ]  0  0  0  0  0  0  0  0
        //        Q4[ 4 - 4 ]  1  0  0  0  0  0  0  0
        //        Q5[ 5 - 4 ]  1  1  0  0  0  0  0  0
        //        Q6[ 6 - 4 ]  1  1  1  0  0  0  0  0
        //        Q7[ 7 - 4 ]  1  1  1  1  0  0  0  0
        //       where 1 at [i, j] means that j token should be forgotten as it can't fit into the sliding
        //       window from the left of i-th token.
        //
        // -   "greater" forms a similar to self-attention mask:
        //
        //        row range [0,... mask_len] > column range [current_pos_id,
        //                                                   ...,
        //                                                   mask_len]
        //
        //       forming, under example conditions, the mask below:
        //        past tokens = 3
        //        present tokens = 5 (starting with current_pos_id = 3)
        //                K0 K1 K2 K3 K4 K5 K6 K7
        //               [ 0  1  2  3  4  5  6  7 ]
        //        Q3[ 3 ]  0  0  0  0  1  1  1  1
        //        Q4[ 4 ]  0  0  0  0  0  1  1  1
        //        Q5[ 5 ]  0  0  0  0  0  0  1  1
        //        Q6[ 6 ]  0  0  0  0  0  0  0  1
        //        Q7[ 7 ]  0  0  0  0  0  0  0  0
        //       where 1 at [i, j] means that j token is a future token for i-th token, that we shouldn't attend to.
        //
        // Together, via "bitwise_or" this two masks forms the inverted sliding attention mask:
        //        past tokens = 3
        //        present tokens = 5 (starting with current_pos_id = 3)
        //        sliding window len = 4
        //                    K0 K1 K2 K3 K4 K5 K6 K7
        //                   [ 0  1  2  3  4  5  6  7 ]
        //        Q3[ 3 - 4 ]  0  0  0  0  1  1  1  1
        //        Q4[ 4 - 4 ]  1  0  0  0  0  1  1  1
        //        Q5[ 5 - 4 ]  1  1  0  0  0  0  1  1
        //        Q6[ 6 - 4 ]  1  1  1  0  0  0  0  1
        //        Q7[ 7 - 4 ]  1  1  1  1  0  0  0  0
        //
        // Issue with sliding attention mask appears when we work with static shapes and different
        // paddings for past and present tokens.
        // More precisely, issue appears with sliding window mask, as Q column range is created
        // from length of past key/values tensor (2175 for 2K case) as start point and the length
        // of attention mask (2176 for 2K) as an end point. This is okay for inverted
        // self-attention mask by means of "greater" operation, as our present tokens exactly
        // left-padded and located on the right in the attention mask.
        // However, for the sliding window mask created by means of "less_equal" operation, given
        // Q range will behave as if position ids of new Q tokens will start from 2175 and not from
        // 3 as in example above and therefore, 2175 - 2047 = 128 first tokens should be forgotten.
        // To fix it a new formula is suggested:
        // 1. (K range <= (Q_pos range - sliding window).T) | (K range > Q range.T)
        // 2. (K range <= (Q range - sliding window).T) & (K range >= len(past_key_values))
        // 3. Resulting mask = 1 | 2,
        // where K range and Q range are created by the same rules as before and Q_pos range is
        // a position_ids array.
        // 4. We also clean mask in places where paddings used instead of real tokens via:
        //    Clean mask = 3 | !(attention_mask_input[past_kv_len:]).T
        auto past_kv_len = opp::wrap_type<ov::op::v8::Gather>({opp::any_input(), opp::any_input(), opp::any_input()});
        auto pos_ids_param = opp::wrap_type<ov::op::v0::Parameter>();
        auto pos_ids_shape_of = opp::wrap_type<ov::op::v3::ShapeOf>({pos_ids_param});
        auto pos_ids_len = opp::wrap_type<ov::op::v8::Gather>({pos_ids_shape_of, opp::any_input(), opp::any_input()});
        auto full_ctx_len = opp::wrap_type<ov::op::v1::Add>({past_kv_len, pos_ids_len});
        auto query_range = opp::wrap_type<ov::op::v4::Range>({past_kv_len, full_ctx_len, opp::any_input()});
        auto column_shape = opp::wrap_type<ov::op::v0::Constant>();
        auto query_range_column = opp::wrap_type<ov::op::v1::Reshape>({query_range, column_shape});

        auto zero_const = opp::wrap_type<ov::op::v0::Constant>();
        auto atten_mask_param = opp::wrap_type<ov::op::v0::Parameter>();
        auto atten_mask_shape_of = opp::wrap_type<ov::op::v3::ShapeOf>({atten_mask_param});
        auto atten_mask_len =
            opp::wrap_type<ov::op::v8::Gather>({atten_mask_shape_of, opp::any_input(), opp::any_input()});
        auto key_range = opp::wrap_type<ov::op::v4::Range>({zero_const, atten_mask_len, opp::any_input()});
        auto key_range_i64 = opp::wrap_type<ov::op::v0::Convert>({key_range});
        auto key_range_f32 = opp::wrap_type<ov::op::v0::Convert>({key_range_i64});

        auto neg_window_size = opp::wrap_type<ov::op::v0::Constant>();
        auto query_left_bound_range = opp::wrap_type<ov::op::v1::Add>({query_range_column, neg_window_size});
        // False in mask means that we shouldn't forget this token
        auto forget_left_tokens_mask = opp::wrap_type<ov::op::v1::LessEqual>({key_range_f32, query_left_bound_range});
        // Basically it is a reference triangle self-attention mask that
        // forbids tokens to attend to future ones, but values are inverted:
        auto look_only_future_mask = opp::wrap_type<ov::op::v1::Greater>({key_range_f32, query_range_column});

        auto inv_sliding_attention_mask =
            opp::wrap_type<ov::op::v13::BitwiseOr>({look_only_future_mask, forget_left_tokens_mask});

        auto callback = [=](opp::Matcher& m) {
            LOG_INFO("Found (4.51) pattern for Phi-3 Sliding Window Attention, will be replaced with custom for static "
                     "shapes.");
            auto& node_to_output = m.get_pattern_value_map();
            auto node_past_kv_len = node_to_output.at(past_kv_len).get_node_shared_ptr();
            auto node_pos_ids_param = node_to_output.at(pos_ids_param).get_node_shared_ptr();
            auto node_atten_mask_param = node_to_output.at(atten_mask_param).get_node_shared_ptr();
            auto node_atten_mask_len = node_to_output.at(atten_mask_len).get_node_shared_ptr();
            auto node_key_range_f32 = node_to_output.at(key_range_f32).get_node_shared_ptr();
            auto node_neg_window_size = node_to_output.at(neg_window_size).get_node_shared_ptr();
            auto node_forget_left_tokens_mask = node_to_output.at(forget_left_tokens_mask).get_node_shared_ptr();
            auto node_bitwise_or = node_to_output.at(inv_sliding_attention_mask).get_node_shared_ptr();

            auto matched_past_kv_len = std::static_pointer_cast<ov::op::v8::Gather>(node_past_kv_len);
            auto matched_pos_ids_input = std::static_pointer_cast<ov::op::v0::Parameter>(node_pos_ids_param);
            auto matched_atten_mask_input = std::static_pointer_cast<ov::op::v0::Parameter>(node_atten_mask_param);
            auto matched_atten_mask_len = std::static_pointer_cast<ov::op::v8::Gather>(node_atten_mask_len);
            auto matched_key_range_f32 = std::static_pointer_cast<ov::op::v0::Convert>(node_key_range_f32);
            auto matched_neg_window_size = std::static_pointer_cast<ov::op::v0::Constant>(node_neg_window_size);
            auto matched_forget_left_tokens_mask =
                std::static_pointer_cast<ov::op::v1::LessEqual>(node_forget_left_tokens_mask);
            auto matched_bitwise_or = std::static_pointer_cast<ov::op::v13::BitwiseOr>(node_bitwise_or);
            OPENVINO_ASSERT(matched_neg_window_size->get_output_size() == 1,
                            "Sliding window size constant must be of size 1, but got " +
                                std::to_string(matched_neg_window_size->get_output_size()));

            // 1.(K range <= (Q_pos range - sliding window).T) | (K range > Q range.T)
            auto query_range_as_pos_ids =
                std::make_shared<ov::op::v0::Convert>(matched_pos_ids_input, ov::element::f32);
            std::vector<int64_t> vector_shape{-1, 1};
            auto vector_shape_const =
                std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{2}, vector_shape);
            auto query_range_as_pos_ids_col =
                std::make_shared<ov::op::v1::Reshape>(query_range_as_pos_ids, vector_shape_const, false);
            auto query_range_as_pos_left_bound =
                std::make_shared<ov::op::v1::Add>(query_range_as_pos_ids_col, matched_neg_window_size);
            auto forget_left_mask_for_right_padding =
                std::make_shared<ov::op::v1::LessEqual>(matched_key_range_f32, query_range_as_pos_left_bound);
            matched_bitwise_or->input(1).replace_source_output(forget_left_mask_for_right_padding);

            // 2. (K range <= (Q range - sliding window).T) & (K range >= shape(past_key_values, 2))
            auto past_kv_len_f32 = std::make_shared<ov::op::v0::Convert>(matched_past_kv_len, ov::element::f32);
            auto only_present_tokens_mask =
                std::make_shared<ov::op::v1::GreaterEqual>(matched_key_range_f32, past_kv_len_f32);
            auto bitwise_and =
                std::make_shared<ov::op::v13::BitwiseAnd>(matched_forget_left_tokens_mask, only_present_tokens_mask);

            // 3. Result = 1 | 2
            // Save target inputs first:
            auto target_inputs = matched_bitwise_or->output(0).get_target_inputs();
            auto new_inv_sliding_mask = std::make_shared<ov::op::v13::BitwiseOr>(matched_bitwise_or, bitwise_and);

            // 4. Removing extra padding via : 3 | !(attention_mask_input[past_kv_len:]).T
            std::vector<int64_t> shape_rank_one{1};
            auto shape_rank_one_const =
                std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, shape_rank_one);
            auto past_len_reshaped =
                std::make_shared<ov::op::v1::Reshape>(matched_past_kv_len, shape_rank_one_const, false);
            auto atten_len_reshaped =
                std::make_shared<ov::op::v1::Reshape>(matched_atten_mask_len, shape_rank_one_const, false);
            auto const_one = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, 1);
            auto present_atten_mask = std::make_shared<ov::op::v8::Slice>(matched_atten_mask_input,
                                                                          past_len_reshaped,
                                                                          atten_len_reshaped,
                                                                          const_one,
                                                                          const_one);
            auto present_atten_mask_bool =
                std::make_shared<ov::op::v0::Convert>(present_atten_mask, ov::element::boolean);
            auto inv_present_atten_mask = std::make_shared<ov::op::v1::LogicalNot>(present_atten_mask_bool);
            auto inv_present_atten_mask_col =
                std::make_shared<ov::op::v1::Reshape>(inv_present_atten_mask, vector_shape_const, false);
            auto clean_inv_sliding_mask =
                std::make_shared<ov::op::v13::BitwiseOr>(new_inv_sliding_mask, inv_present_atten_mask_col);
            for (auto&& input : target_inputs) {
                input.replace_source_output(clean_inv_sliding_mask);
            }

            return true;
        };
        register_matcher(std::make_shared<opp::Matcher>(inv_sliding_attention_mask, "OldPhi3SlidingMaskMatcher"),
                         std::move(callback));
    }
};

class Phi3SlidingMaskMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::npuw::patterns::Phi3SlidingMaskMatcher");

    Phi3SlidingMaskMatcher(const std::shared_ptr<ov::Node>& attention_mask_node_ptr,
                           const std::shared_ptr<ov::Node>& position_ids_node_ptr) {
        // Search for the Phi3 sliding mask pattern to extend it to work with right-padded
        // past tokens and left-padded present tokens. Logic to replace pattern is the same
        // as in Phi3SlidingMask rewriter, but adjusted to another set of operations for
        // creation of mask, obtained from transformers 4.53.
        //
        // Fix is a replace of following pattern:
        // 1. (K range > (Q range - sliding window).T) & (K range <= Q range.T)
        // to
        // 1. (K range > (Q_pos range - sliding window).T) & (K range <= Q range.T)
        // 2. (K range > (Q range - sliding window).T) | (K range < len(past_key_values))
        // 3. Resulting mask = 1 & 2,
        // where K range and Q range are created by the same rules as before and Q_pos range is
        // a position_ids array.
        // 4. We also clean mask in places where paddings used instead of real tokens via:
        //    Clean mask = 3 & attention_mask_input[past_kv_len:].T
        auto unsqueeze_sequence = [&](std::shared_ptr<ov::Node> range) {
            auto unsqueeze1 = opp::wrap_type<ov::op::v0::Unsqueeze>({range, opp::any_input()});
            auto unsqueeze2 = opp::wrap_type<ov::op::v0::Unsqueeze>({unsqueeze1, opp::any_input()});
            auto unsqueeze3 = opp::wrap_type<ov::op::v0::Unsqueeze>({unsqueeze2, opp::any_input()});

            return unsqueeze3;
        };

        auto past_kv_len = opp::wrap_type<ov::op::v8::Gather>({opp::any_input(), opp::any_input(), opp::any_input()});
        // TODO: ov::op::v0::Squeeze may accept or not accept optional axes input, so we need to cover both cases in
        // pattern If Squeeze accepts axes it doesn't match with pattern without axes, and if it doesn't accept axes it
        // doesn't match with pattern with axes. So we need to add two branches in pattern to cover both cases.
        auto past_kv_len_squeeze = opp::optional<ov::op::v0::Squeeze>({past_kv_len});
        auto full_ctx_len = opp::wrap_type<ov::op::v1::Add>({past_kv_len_squeeze, opp::any_input()});
        auto query_range = opp::wrap_type<ov::op::v4::Range>({past_kv_len_squeeze, full_ctx_len, opp::any_input()});
        auto query_range_column = unsqueeze_sequence(query_range);

        auto zero_const = opp::wrap_type<ov::op::v0::Constant>();
        auto full_ctx_len_2 = opp::wrap_type<ov::op::v1::Add>({opp::any_input(), past_kv_len_squeeze});
        auto key_range = opp::wrap_type<ov::op::v4::Range>({zero_const, full_ctx_len_2, opp::any_input()});
        auto key_range_row = unsqueeze_sequence(key_range);
        auto opt_key_range_row_f32 = opp::optional<ov::op::v0::Convert>({key_range_row->output(0)});

        auto neg_window_size = opp::wrap_type<ov::op::v0::Constant>();
        auto query_left_bound_range = opp::wrap_type<ov::op::v1::Add>({query_range_column, neg_window_size});
        // True in mask means that we should attend this token
        auto sliding_mask = opp::wrap_type<ov::op::v1::Greater>({opt_key_range_row_f32, query_left_bound_range});
        auto sliding_and_true = opp::wrap_type<ov::op::v13::BitwiseAnd>({opp::any_input(), sliding_mask});
        // Basically it is a reference triangle self-attention mask
        auto causal_mask = opp::wrap_type<ov::op::v1::LessEqual>({opt_key_range_row_f32, query_range_column});

        auto sliding_and_causal_mask = opp::wrap_type<ov::op::v13::BitwiseAnd>({sliding_and_true, causal_mask});

        auto callback = [=](opp::Matcher& m) {
            auto& node_to_output = m.get_pattern_value_map();

            // Extract matched nodes from pattern
            auto optional_squeeze = node_to_output.find(past_kv_len_squeeze);
            auto matched_past_kv_len = optional_squeeze != node_to_output.end()
                                           ? optional_squeeze->second.get_node_shared_ptr()
                                           : node_to_output.at(past_kv_len).get_node_shared_ptr();
            auto matched_full_ctx_len = node_to_output.at(full_ctx_len).get_node_shared_ptr();
            auto matched_neg_window_size = node_to_output.at(neg_window_size).get_node_shared_ptr();
            auto matched_sliding_mask = node_to_output.at(sliding_mask).get_node_shared_ptr();
            auto matched_sliding_and_causal_mask = node_to_output.at(sliding_and_causal_mask).get_node_shared_ptr();

            auto optional_convert = node_to_output.find(opt_key_range_row_f32);
            auto matched_key_range_row = optional_convert != node_to_output.end()
                                             ? optional_convert->second.get_node_shared_ptr()
                                             : node_to_output.at(key_range_row).get_node_shared_ptr();

            // Call shared transformation helper
            rebuild_sliding_window_mask(attention_mask_node_ptr,
                                        position_ids_node_ptr,
                                        matched_past_kv_len,
                                        matched_full_ctx_len,
                                        matched_key_range_row,
                                        matched_neg_window_size,
                                        matched_sliding_mask,
                                        matched_sliding_and_causal_mask,
                                        "Phi-3, Gemma-2, Gemma-3");
            return true;
        };
        register_matcher(std::make_shared<opp::Matcher>(sliding_and_causal_mask, "Phi3SlidingMaskMatcher"),
                         std::move(callback));
    }
};

class Gemma4SlidingMaskMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::npuw::patterns::Gemma4SlidingMaskMatcher");

    Gemma4SlidingMaskMatcher(const std::shared_ptr<ov::Node>& attention_mask_node_ptr,
                             const std::shared_ptr<ov::Node>& position_ids_node_ptr) {
        // Fix Gemma-4 sliding window attention mask for static KV
        // buffers with different paddings for past and present tokens.
        //
        // Gemma-4's generate model computes the Q range (cache_position) as:
        //   q_range      = Range(0, seq_len, 1)       -- seq_len is the statically-compiled Q
        //                                             -- sequence length (1 for standard generate)
        //   cache_pos    = Add(q_range, past_kv_len)  -- Q position in the full KV buffer
        //   q_idx        = Unsqueeze x3(cache_pos)    -- shape [1,1,seq_len,1]
        //
        // Unlike Phi-3's Range(past_kv_len, past_kv_len + seq_len), Gemma-4 uses an
        // explicit Add, causing the pattern in Phi3SlidingMaskMatcher to miss it.
        //
        // With static KV buffers, the Q position >> actual temporal position_id, so
        // SWA layers incorrectly exclude past prefill tokens (distance > sliding_window).
        //
        // Fix is identical to Phi3SlidingMaskMatcher:
        // 1. (K > pos_ids - window) & (K <= Q_range)
        //    -- temporal position_ids for the left window bound; causal mask for present tokens
        // 2. (K > Q_range - window) | (K < past_kv_len)
        //    -- bound present tokens by window size; always allow past tokens
        // 3. Result = 1 & 2
        //    -- correct sliding window + causal mask for past and present tokens
        // 4. Clean = 3 & attention_mask[past_kv_len:].T
        //    -- remove padding artifacts
        auto unsqueeze_sequence = [&](std::shared_ptr<ov::Node> node) {
            auto u1 = opp::wrap_type<ov::op::v0::Unsqueeze>({node, opp::any_input()});
            auto u2 = opp::wrap_type<ov::op::v0::Unsqueeze>({u1, opp::any_input()});
            auto u3 = opp::wrap_type<ov::op::v0::Unsqueeze>({u2, opp::any_input()});
            return u3;
        };

        // K (key) side: Range(0, past_kv_len + seq_len, 1)
        auto past_kv_len = opp::wrap_type<ov::op::v8::Gather>({opp::any_input(), opp::any_input(), opp::any_input()});
        auto past_kv_len_squeeze = opp::optional<ov::op::v0::Squeeze>({past_kv_len});
        auto zero_const = opp::wrap_type<ov::op::v0::Constant>();
        auto full_ctx_len = opp::wrap_type<ov::op::v1::Add>({opp::any_input(), past_kv_len_squeeze});
        auto key_range = opp::wrap_type<ov::op::v4::Range>({zero_const, full_ctx_len, opp::any_input()});
        auto key_range_row = unsqueeze_sequence(key_range);
        auto opt_key_range_f32 = opp::optional<ov::op::v0::Convert>({key_range_row->output(0)});

        // Q (query) side: Gemma4-specific -- cache_pos = Range(0, seq_len) + past_kv_len
        auto q_range = opp::wrap_type<ov::op::v4::Range>({opp::any_input(), opp::any_input(), opp::any_input()});
        auto cache_position = opp::wrap_type<ov::op::v1::Add>({q_range, past_kv_len_squeeze});
        auto query_range_column = unsqueeze_sequence(cache_position);

        // Sliding window & causal masks (positive form: 1 = attend)
        auto neg_window_size = opp::wrap_type<ov::op::v0::Constant>();
        auto query_left_bound = opp::wrap_type<ov::op::v1::Add>({query_range_column, neg_window_size});
        auto sliding_mask = opp::wrap_type<ov::op::v1::Greater>({opt_key_range_f32, query_left_bound});
        auto sliding_and_true = opp::wrap_type<ov::op::v13::BitwiseAnd>({opp::any_input(), sliding_mask});
        auto causal_mask = opp::wrap_type<ov::op::v1::LessEqual>({opt_key_range_f32, query_range_column});
        auto sliding_and_causal_mask = opp::wrap_type<ov::op::v13::BitwiseAnd>({sliding_and_true, causal_mask});

        auto callback = [=](opp::Matcher& m) {
            auto& node_to_output = m.get_pattern_value_map();

            // Extract matched nodes from pattern
            auto optional_squeeze = node_to_output.find(past_kv_len_squeeze);
            auto matched_past_kv_len = optional_squeeze != node_to_output.end()
                                           ? optional_squeeze->second.get_node_shared_ptr()
                                           : node_to_output.at(past_kv_len).get_node_shared_ptr();
            auto matched_full_ctx_len = node_to_output.at(full_ctx_len).get_node_shared_ptr();
            auto matched_neg_window_size = node_to_output.at(neg_window_size).get_node_shared_ptr();
            auto matched_sliding_mask = node_to_output.at(sliding_mask).get_node_shared_ptr();
            auto matched_sliding_and_causal_mask = node_to_output.at(sliding_and_causal_mask).get_node_shared_ptr();

            auto optional_convert = node_to_output.find(opt_key_range_f32);
            auto matched_key_range_row = optional_convert != node_to_output.end()
                                             ? optional_convert->second.get_node_shared_ptr()
                                             : node_to_output.at(key_range_row).get_node_shared_ptr();

            // Call shared transformation helper
            rebuild_sliding_window_mask(attention_mask_node_ptr,
                                        position_ids_node_ptr,
                                        matched_past_kv_len,
                                        matched_full_ctx_len,
                                        matched_key_range_row,
                                        matched_neg_window_size,
                                        matched_sliding_mask,
                                        matched_sliding_and_causal_mask,
                                        "Gemma4");
            return true;
        };
        register_matcher(std::make_shared<opp::Matcher>(sliding_and_causal_mask, "Gemma4SlidingMaskMatcher"),
                         std::move(callback));
    }
};

#ifdef __GNUC__
#    pragma GCC diagnostic pop
#endif

}  // namespace

namespace ov::npuw {

bool SlidingWindowMask::run_on_model(const std::shared_ptr<ov::Model>& model) {
    std::shared_ptr<ov::Node> attention_mask_node_ptr = nullptr;
    std::shared_ptr<ov::Node> position_ids_node_ptr = nullptr;
    for (const auto& i : model->inputs()) {
        if (i.get_any_name() == "attention_mask") {
            attention_mask_node_ptr = i.get_node_shared_ptr();
        }
        if (i.get_any_name() == "position_ids") {
            position_ids_node_ptr = i.get_node_shared_ptr();
        }
    }
    if (attention_mask_node_ptr == nullptr || position_ids_node_ptr == nullptr) {
        return false;
    }

    auto pos_id_shape = position_ids_node_ptr->get_output_tensor(0).get_partial_shape();
    if (pos_id_shape.size() != 2) {
        // FIXME: Qwen2.5 VL/Omni uses 3D position_ids, which can't be directly used
        //        in creation of sliding window mask.
        return false;
    }

    ov::pass::Manager manager;
    manager.set_per_pass_validation(true);
    const auto rewriter = manager.register_pass<ov::pass::GraphRewrite>();
    rewriter->add_matcher<Gemma4SlidingMaskMatcher>(attention_mask_node_ptr, position_ids_node_ptr);
    rewriter->add_matcher<Phi3SlidingMaskMatcher>(attention_mask_node_ptr, position_ids_node_ptr);
    rewriter->add_matcher<OldPhi3SlidingMaskMatcher>();
    return manager.run_passes(model);
}

}  // namespace ov::npuw