// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/sdpa_to_paged_attention.hpp"

#include <algorithm>
#include <cctype>
#include <optional>
#include <set>
#include <unordered_set>

#include "openvino/cc/pass/itt.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/read_value_base.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/common_optimizations/sdpa_fusion.hpp"
#include "transformations/op_conversions/convert_slice_to_strided_slice.hpp"
#include "transformations/common_optimizations/fuse_gated_delta_net.hpp"
#include "transformations/paged_attention/paged_causal_conv1d_fusion.hpp"
#include "transformations/paged_attention/paged_gated_delta_net_fusion.hpp"
#include "transformations/paged_attention/position_ids_replacer.hpp"
#include "transformations/paged_attention/prev_sequence_length_pattern.hpp"
#include "transformations/paged_attention/state_management_pattern.hpp"
#include "transformations/paged_attention/total_sequence_length_pattern.hpp"
#include "transformations/utils/print_model.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov;
using namespace ov::op;
using namespace ov::pass;

namespace {

struct PARepresentationContext {
    ParameterVector model_wide_params;
    std::map<std::string, std::shared_ptr<v0::Parameter>> optional_model_wide_params;
    std::shared_ptr<v0::Parameter> max_context_len;
    std::shared_ptr<ov::Node> processed_input_ids;
    std::shared_ptr<v0::Parameter> position_ids;
    std::shared_ptr<ov::Node> unsqueezed_position_ids;
};

static std::shared_ptr<v0::Parameter> get_parameter(const std::shared_ptr<ov::Model>& model, const std::string& name) {
    for (const auto& input : model->inputs()) {
        if (input.get_names().count(name)) {
            auto param = ov::as_type_ptr<v0::Parameter>(input.get_node_shared_ptr());
            OPENVINO_ASSERT(param,
                            "The model is in the inconsistent state. Found input '",
                            name,
                            "', but couldn't cast it to v0::Parameter.");
            return param;
        }
    }
    return nullptr;
}

static std::shared_ptr<v0::Parameter> create_or_get_named_parameter(const std::shared_ptr<ov::Model>& model,
                                                                    const std::string& name,
                                                                    const ov::element::Type& type,
                                                                    const ov::PartialShape& pshape) {
    if (auto param = get_parameter(model, name)) {
        return param;
    }
    auto param = std::make_shared<v0::Parameter>(type, pshape);
    param->set_friendly_name(name);
    OPENVINO_ASSERT(param->get_output_size() == 1);
    param->get_output_tensor(0).set_names({name});
    model->add_parameters({param});
    return param;
}

std::optional<size_t> parse_layer_index_from_name(const std::string& name, const std::string& prefix) {
    if (name.rfind(prefix, 0) != 0 || name.size() <= prefix.size()) {
        return std::nullopt;
    }
    const std::string index_str = name.substr(prefix.size());
    if (!std::all_of(index_str.begin(), index_str.end(), ::isdigit)) {
        return std::nullopt;
    }
    return static_cast<size_t>(std::stoul(index_str));
}

static constexpr const char* marked_for_paged_extensions_cleanup = "marked_for_paged_extensions_cleanup";
static constexpr const char* paged_conv_cache_source = "paged_conv_cache_source";

static void mark_for_paged_extensions_cleanup(const std::shared_ptr<ov::Node>& node) {
    node->get_rt_info()[marked_for_paged_extensions_cleanup] = true;
}

static void mark_for_paged_conv_cache_source(const std::shared_ptr<ov::Node>& node) {
    node->get_rt_info()[paged_conv_cache_source] = true;
}

static bool is_marked_for_paged_extensions_cleanup(const std::shared_ptr<const ov::Node>& node) {
    const auto& rt_info = node->get_rt_info();
    return rt_info.count(marked_for_paged_extensions_cleanup) &&
           rt_info.at(marked_for_paged_extensions_cleanup).as<bool>();
}

static PARepresentationContext prepare_sdpa_to_pa_representation(const std::shared_ptr<ov::Model>& model,
                                                                 bool use_per_layer_block_indices_inputs,
                                                                 bool allow_score_aggregation,
                                                                 bool allow_cache_rotation,
                                                                 bool allow_xattention,
                                                                 bool allow_adaptive_rkv,
                                                                 bool allow_qq_bias) {
    OPENVINO_ASSERT(!model->get_variables().empty(),
                    "Model is supposed to be stateful, cannot perform "
                    "the SDPAToPagedAttention transformation. "
                    "For proper conversion run: optimum-cli export openvino --task text-generation-with-past instead "
                    "of --task text-generation");

    OPENVINO_ASSERT(ov::op::util::has_op_with_type<ov::op::v13::ScaledDotProductAttention>(model),
                    "No ScaledDotProductAttention operation observed in the graph, cannot perform "
                    "the SDPAToPagedAttention transformation.");

    const bool already_prepared = (get_parameter(model, "past_lens") != nullptr);

    PARepresentationContext context;
    context.max_context_len = create_or_get_named_parameter(model, "max_context_len", element::i32, PartialShape{});
    context.model_wide_params = {
        create_or_get_named_parameter(model, "past_lens", element::i32, PartialShape{-1}),
        create_or_get_named_parameter(model, "subsequence_begins", element::i32, PartialShape{-1}),
        create_or_get_named_parameter(model, "block_indices_begins", element::i32, PartialShape{-1}),
    };

    if (!use_per_layer_block_indices_inputs) {
        auto block_indices = create_or_get_named_parameter(model, "block_indices", element::i32, PartialShape{-1});
        context.model_wide_params.insert(context.model_wide_params.begin() + 2, block_indices);
    }

    if (allow_score_aggregation) {
        context.optional_model_wide_params["score_aggregation_window"] =
            create_or_get_named_parameter(model, "score_aggregation_window", element::i32, PartialShape{-1});
    }

    if (allow_cache_rotation) {
        context.optional_model_wide_params["model_rotation_trig_lut"] =
            create_or_get_named_parameter(model, "rotation_trig_lut", element::f32, PartialShape{-1, -1});
    }

    if (allow_xattention) {
        context.optional_model_wide_params["xattention_block_size"] =
            create_or_get_named_parameter(model, "xattention_block_size", element::i32, PartialShape{});
        context.optional_model_wide_params["xattention_stride"] =
            create_or_get_named_parameter(model, "xattention_stride", element::i32, PartialShape{});
    }

    if (allow_adaptive_rkv) {
        context.optional_model_wide_params["adaptive_rkv_start_size"] =
            create_or_get_named_parameter(model, "adaptive_rkv_start_size", element::i32, PartialShape{});
        context.optional_model_wide_params["adaptive_rkv_evictable_sizes"] =
            create_or_get_named_parameter(model, "adaptive_rkv_evictable_sizes", element::i32, PartialShape{-1});
    }

    if (allow_qq_bias) {
        context.optional_model_wide_params["qq_bias"] =
            create_or_get_named_parameter(model, "qq_bias", element::u8, PartialShape{-1});
        context.optional_model_wide_params["qq_bias_begins"] =
            create_or_get_named_parameter(model, "qq_bias_begins", element::i32, PartialShape{-1});
    }

    if (already_prepared) {
        // Graph modifications (input_ids reshape, position_ids replacement, pattern passes) were already
        // applied by a prior PrepareSDPAToPARepresentation run. Rebuild context fields from existing params.
        context.position_ids = get_parameter(model, "position_ids");
        OPENVINO_ASSERT(context.position_ids, "Model was expected to have 'position_ids' parameter after preparation.");
        // Reconstruct unsqueezed_position_ids: it is the sole consumer of position_ids in the graph.
        const auto& pos_targets = context.position_ids->get_output_target_inputs(0);
        OPENVINO_ASSERT(!pos_targets.empty(), "position_ids parameter has no consumers after preparation.");
        context.unsqueezed_position_ids = pos_targets.begin()->get_node()->shared_from_this();
        // Reconstruct processed_input_ids: find input_ids or inputs_embeds and walk to its Unsqueeze consumer.
        std::shared_ptr<v0::Parameter> input_ids_node;
        for (const auto& name : {"input_ids", "inputs_embeds"}) {
            if ((input_ids_node = get_parameter(model, name))) {
                break;
            }
        }
        OPENVINO_ASSERT(input_ids_node, "The model doesn't contain input_ids or input_embeds input. Aborting.");
        const auto& input_ids_targets = input_ids_node->get_output_target_inputs(0);
        OPENVINO_ASSERT(!input_ids_targets.empty(), "input_ids parameter has no consumers after preparation.");
        context.processed_input_ids = input_ids_targets.begin()->get_node()->shared_from_this();
        return context;
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

    const auto input_ids_target_inputs = input_ids_node->get_output_target_inputs(0);
    context.processed_input_ids =
        std::make_shared<v0::Unsqueeze>(input_ids_node, v0::Constant::create(element::i32, Shape{}, {1}));
    for (const auto& target : input_ids_target_inputs) {
        target.replace_source_output(context.processed_input_ids);
    }

    if (auto token_type_ids_param = get_parameter(model, "token_type_ids")) {
        token_type_ids_param->validate_and_infer_types();
        context.optional_model_wide_params["token_type_ids"] = std::move(token_type_ids_param);
    }

    context.position_ids = create_or_get_named_parameter(model, "position_ids", element::i64, PartialShape{-1});
    const auto& position_ids_shape = context.position_ids->get_partial_shape();
    if (position_ids_shape.rank().is_static() && position_ids_shape.rank().get_length() == 2) {
        context.position_ids->set_partial_shape(PartialShape{-1});
    } else if (position_ids_shape.rank().is_static() && position_ids_shape.rank().get_length() == 3) {
        context.position_ids->set_partial_shape(PartialShape{position_ids_shape[0], -1});
    } else if (!(position_ids_shape.rank().is_static() && position_ids_shape.rank().get_length() == 1)) {
        OPENVINO_THROW("Unexpected shape for position_ids input: expected rank 1, 2 or 3, observed ",
                       position_ids_shape.rank().is_static() ? position_ids_shape.rank().get_length() : -1);
    }
    context.position_ids->validate_and_infer_types();

    const auto position_ids_target_inputs = context.position_ids->get_output_target_inputs(0);
    context.unsqueezed_position_ids =
        std::make_shared<v0::Unsqueeze>(context.position_ids, v0::Constant::create(element::i32, Shape{}, {-1}));
    for (const auto& target : position_ids_target_inputs) {
        target.replace_source_output(context.unsqueezed_position_ids);
    }

    ov::pass::Manager manager("Prepare SDPA to PA representation");
    manager.set_per_pass_validation(false);
    manager.register_pass<PrevSequenceLengthPattern>(context.processed_input_ids,
                                                     context.max_context_len,
                                                     context.position_ids);
    manager.register_pass<TotalSequenceLengthPattern>(context.max_context_len);
    manager.register_pass<TotalSequenceLengthPatternQwen>(context.max_context_len);
    manager.register_pass<TotalSequenceLengthPatternCodeGen2>(context.max_context_len);
    manager.register_pass<PositionIDsReplacer>(context.unsqueezed_position_ids);
    manager.register_pass<PositionIDsReplacerQwen>(context.unsqueezed_position_ids);
    manager.register_pass<PositionIDsReplacerCodeGen2>(context.position_ids);
    manager.run_passes(model);

    return context;
}

}  // namespace

ov::pass::SDPAToPagedAttention::SDPAToPagedAttention(bool use_per_layer_block_indices_inputs,
                                                     bool use_score_outputs,
                                                     bool allow_score_aggregation,
                                                     bool allow_cache_rotation,
                                                     bool allow_xattention,
                                                     bool allow_adaptive_rkv,
                                                     bool allow_qq_bias)
    : m_use_per_layer_block_indices_inputs(use_per_layer_block_indices_inputs),
      m_use_score_outputs(use_score_outputs),
      m_allow_score_aggregation(allow_score_aggregation),
      m_allow_cache_rotation(allow_cache_rotation),
      m_allow_xattention(allow_xattention),
      m_allow_adaptive_rkv(allow_adaptive_rkv),
      m_allow_qq_bias(allow_qq_bias) {}

ov::pass::PrepareSDPAToPARepresentation::PrepareSDPAToPARepresentation(bool use_per_layer_block_indices_inputs,
                                                                       bool allow_score_aggregation,
                                                                       bool allow_cache_rotation,
                                                                       bool allow_xattention,
                                                                       bool allow_adaptive_rkv)
    : m_use_per_layer_block_indices_inputs(use_per_layer_block_indices_inputs),
      m_allow_score_aggregation(allow_score_aggregation),
      m_allow_cache_rotation(allow_cache_rotation),
      m_allow_xattention(allow_xattention),
      m_allow_adaptive_rkv(allow_adaptive_rkv) {}

bool ov::pass::PrepareSDPAToPARepresentation::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_MODEL_SCOPE(PrepareSDPAToPARepresentation);
    prepare_sdpa_to_pa_representation(model,
                                      m_use_per_layer_block_indices_inputs,
                                      m_allow_score_aggregation,
                                      m_allow_cache_rotation,
                                      m_allow_xattention,
                                      m_allow_adaptive_rkv,
                                      false);
    return true;
}

bool ov::pass::SDPAToPagedAttention::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_MODEL_SCOPE(SDPAToPagedAttention);

    auto context = prepare_sdpa_to_pa_representation(model,
                                                     m_use_per_layer_block_indices_inputs,
                                                     m_allow_score_aggregation,
                                                     m_allow_cache_rotation,
                                                     m_allow_xattention,
                                                     m_allow_adaptive_rkv,
                                                     m_allow_qq_bias);

    ParameterVector kv_parameters;
    ParameterVector block_indices_inputs_for_each_layer;
    ParameterVector rotated_block_indices_inputs_for_each_layer;
    ParameterVector rotation_deltas_inputs_for_each_layer;
    ParameterVector xattention_threshold_inputs_for_each_layer;
    ParameterVector adaptive_rkv_diversity_block_set_indices_inputs_for_each_layer;
    ParameterVector adaptive_rkv_diversity_block_set_indices_begins_inputs_for_each_layer;
    std::unordered_set<std::string> var_ids_to_remove;

    ResultVector score_results;
    ResultVector adaptive_rkv_diversity_results;

    int layer_index = 0;

    ov::pass::Manager manager("SDPA to PA");
    manager.set_per_pass_validation(false);
    manager.register_pass<StateManagementPattern>(kv_parameters,
                                                  context.model_wide_params,
                                                  layer_index,
                                                  context.max_context_len->output(0),
                                                  block_indices_inputs_for_each_layer,
                                                  score_results,
                                                  m_use_per_layer_block_indices_inputs,
                                                  m_use_score_outputs,
                                                  m_allow_cache_rotation,
                                                  m_allow_score_aggregation,
                                                  m_allow_xattention,
                                                  m_allow_adaptive_rkv,
                                                  m_allow_qq_bias,
                                                  rotated_block_indices_inputs_for_each_layer,
                                                  rotation_deltas_inputs_for_each_layer,
                                                  xattention_threshold_inputs_for_each_layer,
                                                  adaptive_rkv_diversity_block_set_indices_inputs_for_each_layer,
                                                  adaptive_rkv_diversity_block_set_indices_begins_inputs_for_each_layer,
                                                  adaptive_rkv_diversity_results,
                                                  context.optional_model_wide_params,
                                                  model,
                                                  var_ids_to_remove);
    manager.run_passes(model);

    {
        // Fallback marking path: ensures all Assigns linked by variable_id from matched
        // ReadValue nodes are marked even if some were not seen in immediate per-match marking.
        auto sinks = model->get_sinks();
        for (const auto& sink : sinks) {
            if (const auto assign = ov::as_type_ptr<ov::op::util::AssignBase>(sink)) {
                if (var_ids_to_remove.count(assign->get_variable_id())) {
                    mark_for_paged_extensions_cleanup(sink);
                }
            }
        }

        // Mark conv-cache ReadValue nodes that were detected by StateManagementPattern.
        for (const auto& node : model->get_ordered_ops()) {
            const auto rv = ov::as_type_ptr<ov::op::util::ReadValueBase>(node);
            if (!rv) {
                continue;
            }
            if (!var_ids_to_remove.count(rv->get_variable_id())) {
                continue;
            }
            if (rv->get_variable_id().find(".past.conv.") == std::string::npos) {
                continue;
            }
            mark_for_paged_conv_cache_source(rv);
        }
    }

    if (m_use_per_layer_block_indices_inputs) {
        model->add_parameters(block_indices_inputs_for_each_layer);
    }

    if (m_use_score_outputs) {
        model->add_results(score_results);
    }

    if (m_allow_score_aggregation) {
        if (!get_parameter(model, "score_aggregation_window")) {
            model->add_parameters({context.optional_model_wide_params["score_aggregation_window"]});
        }
    }

    if (m_allow_cache_rotation) {
        model->add_parameters(rotated_block_indices_inputs_for_each_layer);
        model->add_parameters(rotation_deltas_inputs_for_each_layer);
        if (!get_parameter(model, "rotation_trig_lut")) {
            model->add_parameters({context.optional_model_wide_params["model_rotation_trig_lut"]});
        }
    }

    if (m_allow_xattention) {
        model->add_parameters(xattention_threshold_inputs_for_each_layer);
        if (!get_parameter(model, "xattention_block_size")) {
            model->add_parameters({context.optional_model_wide_params["xattention_block_size"]});
        }
        if (!get_parameter(model, "xattention_stride")) {
            model->add_parameters({context.optional_model_wide_params["xattention_stride"]});
        }
    }
    if (m_allow_adaptive_rkv) {
        if (!get_parameter(model, "adaptive_rkv_start_size")) {
            model->add_parameters({context.optional_model_wide_params["adaptive_rkv_start_size"]});
        }
        if (!get_parameter(model, "adaptive_rkv_evictable_sizes")) {
            model->add_parameters({context.optional_model_wide_params["adaptive_rkv_evictable_sizes"]});
        }
        model->add_parameters(adaptive_rkv_diversity_block_set_indices_inputs_for_each_layer);
        model->add_parameters(adaptive_rkv_diversity_block_set_indices_begins_inputs_for_each_layer);
        model->add_results(adaptive_rkv_diversity_results);
    }

    if (m_allow_qq_bias) {
        if (!get_parameter(model, "qq_bias")) {
            model->add_parameters({context.optional_model_wide_params["qq_bias"]});
        }
        if (!get_parameter(model, "qq_bias_begins")) {
            model->add_parameters({context.optional_model_wide_params["qq_bias_begins"]});
        }
    }

    model->add_parameters(kv_parameters);
    PagedCausalConv1DFusion().run_on_model(model);
    GatedDeltaNetFusion().run_on_model(model);
    PagedGatedDeltaNetFusion().run_on_model(model);
    model->validate_nodes_and_infer_types();

    PagedExtensionsPostCleanup().run_on_model(model);

    return true;
}

ov::pass::PagedExtensionsPostCleanup::PagedExtensionsPostCleanup() {
    set_property(ov::pass::PassProperty::REQUIRE_STATIC_SHAPE, false);
}

bool ov::pass::PagedExtensionsPostCleanup::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_MODEL_SCOPE(PagedExtensionsPostCleanup);

    static constexpr const char* conv_state_table_prefix = "conv_state_table.";
    static constexpr const char* cache_params_present_conv_prefix = "cache_params.present.conv.";

    std::set<size_t> fused_layer_indices;
    for (const auto& parameter : model->get_parameters()) {
        for (const auto& name : parameter->output(0).get_names()) {
            if (const auto layer_index = parse_layer_index_from_name(name, conv_state_table_prefix)) {
                fused_layer_indices.insert(*layer_index);
            }
        }

        if (const auto layer_index =
                parse_layer_index_from_name(parameter->get_friendly_name(), conv_state_table_prefix)) {
            fused_layer_indices.insert(*layer_index);
        }
    }

    for (const auto& result : model->get_results()) {
        for (const auto& name : result->input_value(0).get_names()) {
            if (const auto layer_index = parse_layer_index_from_name(name, cache_params_present_conv_prefix)) {
                fused_layer_indices.insert(*layer_index);
            }
        }
    }

    bool changed = false;

    if (!fused_layer_indices.empty()) {
        ov::ResultVector results_to_remove;
        for (const auto& result : model->get_results()) {
            bool remove_result = false;
            for (const auto& name : result->input_value(0).get_names()) {
                for (const auto layer_index : fused_layer_indices) {
                    const std::string present_name =
                        std::string(cache_params_present_conv_prefix) + std::to_string(layer_index);
                    if (name == present_name) {
                        remove_result = true;
                        break;
                    }
                }
                if (remove_result) {
                    break;
                }
            }
            if (remove_result) {
                results_to_remove.push_back(result);
            }
        }
        for (const auto& result : results_to_remove) {
            model->remove_result(result);
            changed = true;
        }
    }

    ov::ResultVector marked_results_to_remove;
    for (const auto& result : model->get_results()) {
        if (is_marked_for_paged_extensions_cleanup(result)) {
            marked_results_to_remove.push_back(result);
        }
    }
    for (const auto& result : marked_results_to_remove) {
        model->remove_result(result);
        changed = true;
    }

    // Common cleanup for Assign sinks marked by paged transformations.
    ov::SinkVector sinks_to_remove;
    for (const auto& sink : model->get_sinks()) {
        const auto assign_v6 = ov::as_type_ptr<ov::op::v6::Assign>(sink);
        const auto assign_v3 = ov::as_type_ptr<ov::op::v3::Assign>(sink);
        if ((assign_v6 || assign_v3) && is_marked_for_paged_extensions_cleanup(sink)) {
            sinks_to_remove.push_back(sink);
        }
    }
    for (const auto& sink : sinks_to_remove) {
        model->remove_sink(sink);
        changed = true;
    }

    // Remove parameters that are not reachable from model outputs/sinks.
    // This handles dead branches left after paged rewrites where a parameter may still
    // have direct consumers, but the whole subgraph is disconnected from observable outputs.
    std::set<std::shared_ptr<ov::Node>> live_nodes;
    std::vector<std::shared_ptr<ov::Node>> dfs_stack;
    for (const auto& result : model->get_results()) {
        dfs_stack.push_back(result);
    }
    for (const auto& sink : model->get_sinks()) {
        dfs_stack.push_back(sink);
    }

    while (!dfs_stack.empty()) {
        const auto node = dfs_stack.back();
        dfs_stack.pop_back();
        if (!live_nodes.insert(node).second) {
            continue;
        }
        for (const auto& input : node->input_values()) {
            dfs_stack.push_back(input.get_node_shared_ptr());
        }
    }

    ov::ParameterVector params_to_remove;
    for (const auto& parameter : model->get_parameters()) {
        if (live_nodes.count(parameter) > 0) {
            continue;
        }
        params_to_remove.push_back(parameter);
    }
    for (const auto& parameter : params_to_remove) {
        if (model->get_parameter_index(parameter) >= 0) {
            model->remove_parameter(parameter);
            changed = true;
        }
    }

    return changed;
}
