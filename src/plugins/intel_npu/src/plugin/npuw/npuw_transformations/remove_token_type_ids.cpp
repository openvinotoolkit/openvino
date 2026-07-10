// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "remove_token_type_ids.hpp"

#include "../llm_compiled_model_utils.hpp"
#include "../logging.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/openvino.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/validate.hpp"

namespace opp = ov::pass::pattern;

namespace {

// diagnostics warnings on OPENVINO_MATCHER_PASS_RTTI() definition: visibility hidden
#ifdef __GNUC__
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wattributes"
#endif

class RemoveTTIVisionSubgraph : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::npuw::patterns::RemoveTTIVisionSubgraph");

    RemoveTTIVisionSubgraph() {
        auto token_type_ids = opp::wrap_type<ov::op::v0::Parameter>();
        // `token_type_ids` vision subgraph actually can be split to two subgraphs:
        //    - The first part of vision subgraph from `token_type_ids` parameter is used to create a mask for
        //      blockwise attention to image tokens the same as in transformers library:
        //      `get_block_sequence_ids_for_mask()` from
        //      https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma3/modeling_gemma3.py.
        //    - The second part of vision subgraph from `token_type_ids` parameter also handles image tokens in
        //      a way to find them inside given parameter and select according result of `Less` operation.
        //      Result of second subgraph is used to make BitwiseAND with the created above blockwise mask.
        auto subg1_equal = opp::wrap_type<ov::op::v1::Equal>({token_type_ids, opp::any_input()});
        auto subg1_pad = opp::wrap_type<ov::op::v1::Pad, ov::op::v12::Pad>(
            {subg1_equal, opp::any_input(), opp::any_input(), opp::any_input()});
        auto subg1_slice = opp::wrap_type<ov::op::v8::Slice>(
            {subg1_pad, opp::any_input(), opp::any_input(), opp::any_input(), opp::any_input()});
        auto subg1_bw_not = opp::wrap_type<ov::op::v13::BitwiseNot>({subg1_slice});
        auto subg1_bw_and = opp::wrap_type<ov::op::v13::BitwiseAnd>({subg1_equal, subg1_bw_not});
        auto subg1_convert = opp::wrap_type<ov::op::v0::Convert>({subg1_bw_and});
        auto subg1_cum_sum = opp::wrap_type<ov::op::v0::CumSum>({subg1_convert, opp::any_input()});
        auto subg1_add = opp::wrap_type<ov::op::v1::Add>({subg1_cum_sum, opp::any_input()});
        auto subg1_convert_add = opp::wrap_type<ov::op::v0::Convert>({subg1_add});
        auto subg1_select = opp::wrap_type<ov::op::v1::Select>({subg1_equal, subg1_convert_add, opp::any_input()});

        // We are taking only `ShapeOf` path from the first subgraph, as it is enough for us to find the
        // correct merging point with the second subgraph and the Sliding Causal/Causal mask from it.
        auto subg1_shape_of = opp::wrap_type<ov::op::v0::ShapeOf, ov::op::v3::ShapeOf>({subg1_select});
        auto subg1_gather = opp::wrap_type<ov::op::v8::Gather>({subg1_shape_of, opp::any_input(), opp::any_input()});
        auto subg1_less = opp::wrap_type<ov::op::v1::Less>({opp::any_input(), subg1_gather});

        // In dynamic version of the model, there is branching to two paths after `Less` op:
        // one path is for local attention and another one is for global attention.
        // Both path utilize the same `Select`->`Equal`subgraph below, so pattern should match twice.
        auto subg1_branch_select = opp::wrap_type<ov::op::v1::Select>({subg1_less, opp::any_input(), opp::any_input()});
        auto subg1_branch_equal = opp::wrap_type<ov::op::v1::Equal>({opp::any_input(), subg1_branch_select});

        auto subg2_reshape = opp::wrap_type<ov::op::v1::Reshape>({token_type_ids, opp::any_input()});

        // There is branching point for local and global attention after `Reshape` in the second subgraph.
        // Both local and global attentions path utilize the same subgraph with two `Gather`-s below.
        auto subg2_branch_gather1 =
            opp::wrap_type<ov::op::v8::Gather>({subg2_reshape, opp::any_input(), opp::any_input()});
        auto subg2_branch_gthr1_reshape = opp::wrap_type<ov::op::v1::Reshape>({subg2_branch_gather1, opp::any_input()});
        auto subg2_branch_gthr1_reshape_reshape =
            opp::wrap_type<ov::op::v1::Reshape>({subg2_branch_gthr1_reshape, opp::any_input()});
        auto subg2_branch_gthr1_equal =
            opp::wrap_type<ov::op::v1::Equal>({subg2_branch_gthr1_reshape_reshape, opp::any_input()});
        auto subg2_branch_gather2 =
            opp::wrap_type<ov::op::v8::Gather>({subg2_reshape, opp::any_input(), opp::any_input()});
        auto subg2_branch_gthr2_reshape = opp::wrap_type<ov::op::v1::Reshape>({subg2_branch_gather2, opp::any_input()});
        auto subg2_branch_gthr2_reshape_reshape =
            opp::wrap_type<ov::op::v1::Reshape>({subg2_branch_gthr2_reshape, opp::any_input()});
        auto subg2_branch_gthr2_select = opp::wrap_type<ov::op::v1::Select>(
            {opp::any_input(), subg2_branch_gthr2_reshape_reshape, opp::any_input()});
        auto subg2_branch_gthr2_equal =
            opp::wrap_type<ov::op::v1::Equal>({subg2_branch_gthr2_select, opp::any_input()});
        auto subg2_branch_bw_and =
            opp::wrap_type<ov::op::v13::BitwiseAnd>({subg2_branch_gthr1_equal, subg2_branch_gthr2_equal});

        // Vision block is the subgraph that will be passed to BitwiseOR with Causal mask or Causal Sliding mask futher.
        auto branch_vision_block = opp::wrap_type<ov::op::v13::BitwiseAnd>({subg2_branch_bw_and, subg1_branch_equal});
        auto branch_causal_or_vision = opp::wrap_type<ov::op::v13::BitwiseOr>({opp::any_input(), branch_vision_block});

        auto callback = [=](opp::Matcher& m) {
            auto& node_to_output = m.get_pattern_value_map();
            auto matched_token_type_ids = node_to_output.at(token_type_ids).get_node_shared_ptr();
            if (matched_token_type_ids->output(0).get_names().count(std::string(ov::npuw::token_type_ids_name)) == 0) {
                return false;
            }
            auto matched_causal_or_vision = node_to_output.at(branch_causal_or_vision).get_node_shared_ptr();
            auto zero = std::make_shared<ov::op::v0::Constant>(ov::element::boolean, ov::Shape{}, false);
            matched_causal_or_vision->input(1).replace_source_output(zero);
            return true;
        };
        register_matcher(std::make_shared<opp::Matcher>(branch_causal_or_vision, "RemoveTTIVisionSubgraph"),
                         std::move(callback));
    }
};

class RemoveTTIShapeOfSubgraph : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::npuw::patterns::RemoveTTIShapeOfSubgraph");

    RemoveTTIShapeOfSubgraph() {
        auto token_type_ids = opp::wrap_type<ov::op::v0::Parameter>();
        // There is a subgraph from `token_type_ids` parameter which ends at ShapeOf operation, that in its turn,
        // is passed to Reshape of `attention_mask` operations chain.
        // Later on, result of that reshaped `attention_mask` will go to BitwiseAND with mask from BitwiseOR(Causal
        // mask, Vision mask). Such subgraph from `token_type_ids` should be carefully disconnected from
        // `attention_mask` components and removed to clean all dependencies.
        auto tti_shape_of = opp::wrap_type<ov::op::v0::ShapeOf, ov::op::v3::ShapeOf>({token_type_ids});
        auto tti_gather = opp::wrap_type<ov::op::v8::Gather>({tti_shape_of, opp::any_input(), opp::any_input()});
        auto tti_less = opp::wrap_type<ov::op::v1::Less>({opp::any_input(), tti_gather});
        auto tti_select = opp::wrap_type<ov::op::v1::Select>({tti_less, opp::any_input(), opp::any_input()});
        auto tti_convert = opp::wrap_type<ov::op::v0::Convert>({tti_select});
        auto tti_add = opp::wrap_type<ov::op::v1::Add>({tti_convert, opp::any_input()});
        auto tti_shape_of_2 = opp::wrap_type<ov::op::v0::ShapeOf, ov::op::v3::ShapeOf>({tti_add});

        auto attention_mask = opp::wrap_type<ov::op::v0::Parameter>();
        auto attn_convert = opp::wrap_type<ov::op::v0::Convert>({attention_mask});
        auto attn_reshape = opp::wrap_type<ov::op::v1::Reshape>({attn_convert, opp::any_input()});

        // Here we face a branching point for local and global attention
        auto branch_attn_gather =
            opp::wrap_type<ov::op::v8::Gather>({attn_reshape, opp::any_input(), opp::any_input()});
        auto branch_attn_reshape_2 = opp::wrap_type<ov::op::v1::Reshape>({branch_attn_gather, opp::any_input()});
        auto branch_attn_tti_reshape = opp::wrap_type<ov::op::v1::Reshape>({branch_attn_reshape_2, tti_shape_of_2});
        auto branch_causal_or_vis_and_reshape =
            opp::wrap_type<ov::op::v13::BitwiseAnd>({opp::any_input(), branch_attn_tti_reshape});

        auto callback = [=](opp::Matcher& m) {
            auto& node_to_output = m.get_pattern_value_map();
            auto matched_token_type_ids = node_to_output.at(token_type_ids).get_node_shared_ptr();
            if (matched_token_type_ids->output(0).get_names().count(std::string(ov::npuw::token_type_ids_name)) == 0) {
                return false;
            }

            // In models without `token_type_ids` the shape (for `attention_mask`) is taken
            // from the same `Add` operation which feeds the preceding `Gather->Reshape` subgraph.
            // This `Add` operation is passed as the second (indices) argument to `Gather`.
            auto matched_tti_shape_of_2 = node_to_output.at(tti_shape_of_2).get_node_shared_ptr();
            auto matched_branch_attn_gather = node_to_output.at(branch_attn_gather).get_node_shared_ptr();
            matched_tti_shape_of_2->input(0).replace_source_output(
                matched_branch_attn_gather->input(1).get_source_output());
            return true;
        };
        register_matcher(std::make_shared<opp::Matcher>(branch_causal_or_vis_and_reshape, "RemoveTTIShapeOfSubgraph"),
                         std::move(callback));
    }
};

#ifdef __GNUC__
#    pragma GCC diagnostic pop
#endif
}  // anonymous namespace

bool ov::npuw::RemoveTokenTypeIds::run_on_model(const std::shared_ptr<ov::Model>& model) {
    if (ov::npuw::util::has_input(model, token_type_ids_name) == false) {
        return false;
    }
    // For Gemma3 generate model, we need to remove blockwise mask created from `token_type_ids` parameter
    // as well as second subgraph from `token_type_ids` parameter, that is used to create a reshape for the
    // `attention_mask`. These transformations are needed to avoid accuracy issues due to incorrect interaction
    // of created vision mask with static shapes and different paddings and to allow full removal of
    // `token_type_ids` parameter from the model to skip unnecessary operations with it.
    // As `token_type_ids` isn't used in the generate stage, it is safe to remove the subgraphs.
    // However, components and connections in `attention_mask` subgraph should be fully preserved.
    //
    // Apply both passes independently and require BOTH to succeed before removing parameter.
    ov::pass::Manager vision_manager("remove-token-type-ids-vision");
    vision_manager.set_per_pass_validation(false);
    vision_manager.register_pass<RemoveTTIVisionSubgraph>();
    auto vision_removed = vision_manager.run_passes(model);

    ov::pass::Manager shapeof_manager("remove-token-type-ids-shapeof");
    shapeof_manager.set_per_pass_validation(false);
    shapeof_manager.register_pass<RemoveTTIShapeOfSubgraph>();
    auto shapeof_removed = shapeof_manager.run_passes(model);

    bool both_subgraphs_removed = vision_removed && shapeof_removed;
    
    if (both_subgraphs_removed) {
        LOG_INFO("RemoveTokenTypeIds: both vision and shapeof subgraphs were found and removed. Removing `token_type_ids` parameter.");
        auto token_type_ids_param =
            ov::as_type_ptr<ov::op::v0::Parameter>(model->input(token_type_ids_name).get_node_shared_ptr());
        model->remove_parameter(token_type_ids_param);
        model->validate_nodes_and_infer_types();
    } else if (vision_removed || shapeof_removed) {
        LOG_WARN("RemoveTokenTypeIds: only partial subgraphs removed (vision=%d, shapeof=%d). Parameter not removed.",
                 vision_removed, shapeof_removed);
    } else {
        LOG_WARN("RemoveTokenTypeIds: `token_type_ids` exists but subgraphs were not found in generate model.");
    }

    return both_subgraphs_removed;
}
