// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/sdpa_to_paged_attention.hpp"

#include "openvino/cc/pass/itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/common_optimizations/fuse_gated_delta_net.hpp"
#include "transformations/common_optimizations/sdpa_fusion.hpp"
#include "transformations/op_conversions/convert_slice_to_strided_slice.hpp"
#include "transformations/paged_attention/attention_mask_shape_replacer.hpp"
#include "transformations/paged_attention/eliminate_conv_padding_mask_gating.hpp"
#include "transformations/paged_attention/paged_causal_conv1d_fusion.hpp"
#include "transformations/paged_attention/paged_gated_delta_net_fusion.hpp"
#include "transformations/paged_attention/position_ids_replacer.hpp"
#include "transformations/paged_attention/prev_sequence_length_pattern.hpp"
#include "transformations/paged_attention/state_management_pattern.hpp"
#include "transformations/paged_attention/total_sequence_length_pattern.hpp"
#include "transformations/utils/print_model.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov::op;
using ov::pass::paged_attention::PaParams;
using ov::pass::paged_attention::PaResults;

namespace {

std::shared_ptr<v0::Parameter> get_parameter(const std::shared_ptr<ov::Model>& model, const std::string& name) {
    for (const auto& param : model->inputs()) {
        const auto& names = param.get_names();
        if (names.count(name)) {
            if (auto casted_param = ov::as_type_ptr<v0::Parameter>(param.get_node_shared_ptr())) {
                return casted_param;
            } else {
                OPENVINO_THROW("The model is in the inconsistent state. Found input '",
                               name,
                               "', but couldn't cast it to v0::Parameter.");
            }
        }
    }

    return nullptr;
}

}  // namespace

ov::pass::SDPAToPagedAttention::SDPAToPagedAttention(bool use_per_layer_block_indices_inputs,
                                                     bool use_score_outputs,
                                                     bool allow_score_aggregation,
                                                     bool allow_cache_rotation,
                                                     bool allow_xattention,
                                                     bool allow_adaptive_rkv,
                                                     bool allow_qq_bias)
    : m_options{use_per_layer_block_indices_inputs,
                use_score_outputs,
                allow_score_aggregation,
                allow_cache_rotation,
                allow_xattention,
                allow_adaptive_rkv,
                allow_qq_bias} {}

bool ov::pass::SDPAToPagedAttention::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_MODEL_SCOPE(SDPAToPagedAttention);

    OPENVINO_ASSERT(!model->get_variables().empty(),
                    "Model is supposed to be stateful, cannot perform "
                    "the SDPAToPagedAttention transformation. "
                    "For proper conversion run: optimum-cli export openvino --task text-generation-with-past instead "
                    "of --task text-generation");

    OPENVINO_ASSERT(ov::op::util::has_op_with_type<ov::op::v13::ScaledDotProductAttention>(model),
                    "No ScaledDotProductAttention operation observed in the graph, cannot perform "
                    "the SDPAToPagedAttention transformation.");

    m_params = PaParams{model->get_parameters()};
    m_results = PaResults{model->get_results()};
    auto max_context_len = m_params.add("max_context_len", element::i32, PartialShape{});
    m_params.add("past_lens", element::i32, PartialShape{-1});
    m_params.add("subsequence_begins", element::i32, PartialShape{-1});
    m_params.add("block_indices_begins", element::i32, PartialShape{-1});
    if (!m_options.use_per_layer_block_indices_inputs) {
        m_params.add("block_indices", element::i32, PartialShape{-1});
    }

    std::shared_ptr<v0::Parameter> input_ids_node;
    for (const auto& name : {"input_ids", "inputs_embeds"}) {
        if ((input_ids_node = get_parameter(model, name))) {
            break;
        }
    }

    OPENVINO_ASSERT(input_ids_node, "The model doesn't contain input_ids or input_embeds input. Aborting.");

    if (input_ids_node->get_friendly_name() == "input_ids") {
        input_ids_node->set_partial_shape(PartialShape{-1});
    } else if (input_ids_node->get_friendly_name() == "inputs_embeds") {
        input_ids_node->set_partial_shape(PartialShape{-1, -1});
    }

    auto input_ids_target_inputs = input_ids_node->get_output_target_inputs(0);
    auto processed_input_ids =
        std::make_shared<v0::Unsqueeze>(input_ids_node, v0::Constant::create(element::i32, Shape{}, {1}));
    for (const auto& target : input_ids_target_inputs) {
        target.replace_source_output(processed_input_ids);
    }

    std::unordered_set<std::string> var_ids_to_remove;

    // Get-or-create the flattened position_ids and restore its rank at each existing consumer. The
    // PositionIDsReplacer* passes below consume the raw parameter and add their own rank-restoring nodes.
    auto position_ids = paged_attention::prepare_position_ids(m_params);

    ov::pass::Manager manager("SDPA to PA");
    manager.set_per_pass_validation(false);
    manager.register_pass<ov::pass::GatedDeltaNetFusion>();  // This pass is required to ensure that all GatedDeltaNet
                                                             // nodes are in the expected form before running
                                                             // PagedGatedDeltaNetFusion.
    manager.register_pass<StateManagementPattern>(m_params, m_results, m_options, var_ids_to_remove);
    manager.register_pass<EliminateConvPaddingMaskGating>();
    manager.register_pass<AttentionMaskShapeReplacer>(input_ids_node);
    manager.register_pass<PagedCausalConv1DFusion>(m_params, var_ids_to_remove);
    manager.register_pass<PagedGatedDeltaNetFusion>(m_params, var_ids_to_remove);
    manager.register_pass<PrevSequenceLengthPattern>(processed_input_ids, max_context_len, position_ids);
    manager.register_pass<TotalSequenceLengthPattern>(max_context_len);
    manager.register_pass<TotalSequenceLengthPatternQwen>(max_context_len);
    manager.register_pass<TotalSequenceLengthPatternCodeGen2>(max_context_len);
    manager.register_pass<PositionIDsReplacer>(position_ids);
    manager.register_pass<PositionIDsReplacerQwen>(position_ids);
    manager.register_pass<PositionIDsReplacerCodeGen2>(position_ids);
    manager.register_pass<PositionIDsReplacerLFM2>(position_ids);
    manager.run_passes(model);

    {
        // Remove all Assigns aggressively, the path from the kv-cache concat to Assign can be complicated,
        // but there is no reason to track it and reject part of the Assigns, because the model will remain
        // in incorrect form anyway.
        auto sinks = model->get_sinks();

        for (auto& sink : sinks) {
            if (auto assign = ov::as_type_ptr<ov::op::util::AssignBase>(sink)) {
                if (var_ids_to_remove.count(assign->get_variable_id())) {
                    model->remove_sink(sink);
                }
            }
        }
    }

    for (auto& param_name : {"beam_idx", "attention_mask"}) {
        if (auto param = get_parameter(model, param_name)) {
            model->remove_parameter(param);

            if (param->output(0).get_target_inputs().size() == 0) {
                std::stringstream consumers;
                consumers << std::endl;
                for (auto& input : param->output(0).get_target_inputs()) {
                    consumers << *input.get_node() << std::endl;
                }
                OPENVINO_ASSERT(param->output(0).get_target_inputs().size() == 0,
                                "PagedAttention transformation failed: couldn't remove ",
                                param->output(0).get_target_inputs().size(),
                                " inputs of ",
                                param_name,
                                " input: ",
                                consumers.str());
            }
        }
    }

    model->add_results(m_results.items());
    model->add_parameters(m_params.items());
    model->validate_nodes_and_infer_types();

    return true;
}
