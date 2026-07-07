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
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/validate.hpp"

namespace opp = ov::pass::pattern;

namespace {

// diagnostics warnings on OPENVINO_MATCHER_PASS_RTTI() definition: visibility hidden
#ifdef __GNUC__
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wattributes"
#endif


class RemoveTTISubgraphs : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::npuw::patterns::RemoveTTISubgraphs");

    // For Gemma3 generate model, we need to remove blockwise mask created from `token_type_ids` parameter.
    // As `token_type_ids` isn't used in the generate stage, it is safe to remove them.
    // This avoids accuracy issues due to incorrect interaction of created mask with static shapes and different paddings.
    RemoveTTISubgraphs() {
        auto token_type_ids = opp::wrap_type<ov::op::v0::Parameter>();
        // The subgraph below is created from `token_type_ids` parameter and is used to create a mask for blockwise attention
        // for image tokens.
        auto subg1_equal = opp::wrap_type<ov::op::v1::Equal>({token_type_ids, opp::any_input()});
        auto subg1_pad = opp::wrap_type<ov::op::v1::Pad, ov::op::v12::Pad>({subg1_equal, opp::any_input(), opp::any_input(), opp::any_input()});
        auto subg1_slice = opp::wrap_type<ov::op::v8::Slice>({subg1_pad, opp::any_input(), opp::any_input(), opp::any_input(), opp::any_input()});
        auto subg1_bw_not = opp::wrap_type<ov::op::v13::BitwiseNot>({subg1_slice});
        auto subg1_bw_and = opp::wrap_type<ov::op::v13::BitwiseAnd>({subg1_equal, subg1_bw_not});
        auto subg1_convert = opp::wrap_type<ov::op::v0::Convert>({subg1_bw_and});
        auto subg1_cum_sum = opp::wrap_type<ov::op::v0::CumSum>({subg1_convert, opp::any_input()});
        auto subg1_add = opp::wrap_type<ov::op::v1::Add>({subg1_cum_sum, opp::any_input()});
        auto subg1_convert_add = opp::wrap_type<ov::op::v0::Convert>({subg1_add});
        auto subg1_select = opp::wrap_type<ov::op::v1::Select>({subg1_equal, subg1_convert_add, opp::any_input()});

        auto subg1_shape_of = opp::wrap_type<ov::op::v0::ShapeOf, ov::op::v3::ShapeOf>({subg1_select});
        auto subg1_gather = opp::wrap_type<ov::op::v8::Gather>({subg1_shape_of, opp::any_input(), opp::any_input()});
        auto subg1_less = opp::wrap_type<ov::op::v1::Less>({opp::any_input(), subg1_gather});

        // In dynamic version of the model, there is branching to two paths after `Less` op:
        // one path is for local attention and another one is for global attention.
        // Both path utilize the same subgraph below, so pattern should match twice.
        auto subg1_branch_select = opp::wrap_type<ov::op::v1::Select>({subg1_less, opp::any_input(), opp::any_input()});
        auto subg1_branch_equal = opp::wrap_type<ov::op::v1::Equal>({opp::any_input(), subg1_branch_select});
        // Vision block is the subgraph that will be passed to BitwiseOR with Causal mask or Causal Sliding mask futher.
        auto subg1_branch_vision_block = opp::wrap_type<ov::op::v13::BitwiseAnd>({opp::any_input(), subg1_branch_equal});

        // There is another subgraph from `token_type_ids` parameter that will be passed to BitwiseAND with the resulted mask
        // from BitwiseOR(Causal mask, Vision block). This subgraph should be removed too to
        // clean all dependencies from `token_type_ids` parameter.
        // Note: this subgraph doesn't break accuracy when first one is removed.
        auto subg2_shape_of = opp::wrap_type<ov::op::v0::ShapeOf, ov::op::v3::ShapeOf>({token_type_ids});
        auto subg2_gather = opp::wrap_type<ov::op::v8::Gather>({subg2_shape_of, opp::any_input(), opp::any_input()});
        auto subg2_less = opp::wrap_type<ov::op::v1::Less>({opp::any_input(), subg2_gather});
        auto subg2_select = opp::wrap_type<ov::op::v1::Select>({subg2_less, opp::any_input(), opp::any_input()});
        auto subg2_convert = opp::wrap_type<ov::op::v0::Convert>({subg2_select});
        auto subg2_add = opp::wrap_type<ov::op::v1::Add>({subg2_convert, opp::any_input()});
        auto subg2_shape_of_2 = opp::wrap_type<ov::op::v0::ShapeOf, ov::op::v3::ShapeOf>({subg2_add});

        // Here we face a new branching point for local and global attention, now from the second subgraph.
        auto subg2_branch_reshape = opp::wrap_type<ov::op::v1::Reshape>({opp::any_input(), subg2_shape_of_2});

        // Merge point of the two subgraphs (also exists in two branches: for local and global attentions):
        auto branch_causal_or_blockwise = opp::wrap_type<ov::op::v13::BitwiseOr>({opp::any_input(), subg1_branch_vision_block});
        auto branch_true_and_result = opp::wrap_type<ov::op::v13::BitwiseAnd>({opp::any_input(), branch_causal_or_blockwise});
        auto branch_result_and_subg2 = opp::wrap_type<ov::op::v13::BitwiseAnd>({branch_true_and_result, subg2_branch_reshape});

        auto callback = [=](opp::Matcher& m) {
            auto& node_to_output = m.get_pattern_value_map();
            auto matched_causal_or_vision = node_to_output.at(branch_causal_or_blockwise).get_node_shared_ptr();
            auto matched_final_and = node_to_output.at(branch_result_and_subg2).get_node_shared_ptr();
            auto zero = std::make_shared<ov::op::v0::Constant>(ov::element::boolean, ov::Shape{}, false);
            matched_causal_or_vision->get_input_source_output(1).replace(zero->output(0));
            auto one = std::make_shared<ov::op::v0::Constant>(ov::element::boolean, ov::Shape{}, true);
            matched_final_and->get_input_source_output(1).replace(one->output(0));
            return true;
        };
        register_matcher(std::make_shared<opp::Matcher>(branch_result_and_subg2, "RemoveTTISubgraphs"),
                         std::move(callback));
    }
};

#ifdef __GNUC__
#    pragma GCC diagnostic pop
#endif
}  // anonymous namespace

bool ov::npuw::RemoveTokenTypeIds::run_on_model(const std::shared_ptr<ov::Model>& model) {
    if (ov::npuw::util::has_input(model, "token_type_ids") == false) {
        return false;
    }

    ov::pass::Manager manager("remove-token-type-ids");
    manager.set_per_pass_validation(false);
    manager.register_pass<RemoveTTISubgraphs>();
    auto subgraph_replaced = manager.run_passes(model);
    if (subgraph_replaced) {
        LOG_INFO("RemoveTokenTypeIds: `token_type_ids` subgraphs were found and removed in generate model.");
    } else {
        LOG_WARN("RemoveTokenTypeIds: `token_type_ids` exists butsubgraphs were not found in generate model.");
    }

    auto token_type_ids_param =
        ov::as_type_ptr<ov::op::v0::Parameter>(model->input("token_type_ids").get_node()->shared_from_this());
    model->remove_parameter(token_type_ids_param);
    model->validate_nodes_and_infer_types();
    return true;
}
