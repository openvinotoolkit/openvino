// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/sdpa_to_paged_attention.hpp"

#include "openvino/cc/pass/itt.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/sdpa_to_paged_attention/position_ids_replacer.hpp"
#include "transformations/sdpa_to_paged_attention/prev_sequence_length_pattern.hpp"
#include "transformations/sdpa_to_paged_attention/state_management_pattern.hpp"
#include "transformations/sdpa_to_paged_attention/total_sequence_length_pattern.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov::op;

ov::pass::SDPAToPagedAttention::SDPAToPagedAttention(bool use_per_layer_block_indices_inputs,
                                                     bool use_score_outputs,
                                                     bool allow_cache_rotation)
    : m_use_per_layer_block_indices_inputs(use_per_layer_block_indices_inputs),
      m_use_score_outputs(use_score_outputs),
      m_allow_cache_rotation(allow_cache_rotation) {}

static std::shared_ptr<v0::Parameter> setName(std::shared_ptr<v0::Parameter> node, const char* name) {
    // Set name for both node and output tensor (should be only one tensor, and any other names will be overriden by a
    // given single name)
    node->set_friendly_name(name);
    OPENVINO_ASSERT(node->get_output_size() == 1);
    node->get_output_tensor(0).set_names({name});
    return node;
}

bool ov::pass::SDPAToPagedAttention::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_MODEL_SCOPE(SDPAToPagedAttention);

    OPENVINO_ASSERT(ov::op::util::has_op_with_type<ov::op::v13::ScaledDotProductAttention>(model),
                    "No ScaledDotProductAttention operation observed in the graph, cannot perform "
                    "the SDPAToPagedAttention transformation.");

    auto max_context_len = setName(std::make_shared<v0::Parameter>(element::i32, PartialShape{}), "max_context_len");
    ParameterVector model_remaining_params{
        setName(std::make_shared<v0::Parameter>(element::i32, PartialShape{-1}), "past_lens"),
        setName(std::make_shared<v0::Parameter>(element::i32, PartialShape{-1}), "subsequence_begins"),
        setName(std::make_shared<v0::Parameter>(element::i32, PartialShape{-1}), "block_indices_begins"),
    };
    if (!m_use_per_layer_block_indices_inputs) {
        auto block_indices = setName(std::make_shared<v0::Parameter>(element::i32, PartialShape{-1}), "block_indices");
        model_remaining_params.insert(model_remaining_params.begin() + 2, block_indices);
    }

    std::shared_ptr<v0::Parameter> model_rotation_trig_lut;

    if (m_allow_cache_rotation) {
        model_rotation_trig_lut =
            setName(std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, -1}), "rotation_trig_lut");
    }

    auto sliding_window = v0::Constant::create(element::i32, Shape{}, {0});

    auto get_parameter = [=](const std::shared_ptr<ov::Model>& model,
                             const std::string& name) -> std::shared_ptr<v0::Parameter> {
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
    };

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

    ParameterVector kv_parameters;
    ParameterVector parameters_to_remove;
    ResultVector results_to_remove;  // # used, but cannot really track all Results in stateless model
    ParameterVector block_indices_inputs_for_each_layer;
    ParameterVector rotated_block_indices_inputs_for_each_layer;
    ParameterVector rotation_deltas_inputs_for_each_layer;

    ResultVector score_results;

    std::shared_ptr<v0::Parameter> position_ids;
    if (!get_parameter(model, "position_ids")) {
        position_ids = setName(std::make_shared<v0::Parameter>(element::i64, PartialShape{-1}), "position_ids");
        model->add_parameters({position_ids});
    } else {
        position_ids = ov::as_type_ptr<v0::Parameter>(model->input("position_ids").get_node_shared_ptr());
        position_ids->set_partial_shape(PartialShape{-1});
        position_ids->validate_and_infer_types();
    }
    auto position_ids_target_inputs = position_ids->get_output_target_inputs(0);
    auto unsqueezed_position_ids =
        std::make_shared<v0::Unsqueeze>(position_ids, v0::Constant::create(element::i32, Shape{}, {1}));
    for (const auto& target : position_ids_target_inputs) {
        target.replace_source_output(unsqueezed_position_ids);
    }

    int layer_index = 0;

    ov::pass::Manager manager("SDPA to PA");
    manager.set_per_pass_validation(false);
    manager.register_pass<StateManagementPattern>(kv_parameters,
                                                  model_remaining_params,
                                                  sliding_window,
                                                  parameters_to_remove,
                                                  layer_index,
                                                  max_context_len->output(0),
                                                  block_indices_inputs_for_each_layer,
                                                  score_results,
                                                  m_use_per_layer_block_indices_inputs,
                                                  m_use_score_outputs,
                                                  m_allow_cache_rotation,
                                                  rotated_block_indices_inputs_for_each_layer,
                                                  rotation_deltas_inputs_for_each_layer,
                                                  model_rotation_trig_lut);
    manager.register_pass<PrevSequenceLengthPattern>(processed_input_ids, max_context_len, position_ids);
    manager.register_pass<TotalSequenceLengthPattern>(max_context_len);
    manager.register_pass<TotalSequenceLengthPatternQwen>(max_context_len);
    manager.register_pass<PositionIDsReplacer>(unsqueezed_position_ids);
    manager.register_pass<PositionIDsReplacerQwen>(unsqueezed_position_ids);
    manager.run_passes(model);

    {
        // Remove all Assigns aggressively, the path from the kv-cache concat to Assign can be complicated,
        // but there is no reason to track it and reject part of the Assigns, because the model will remain
        // in incorrect form anyway.
        auto sinks = model->get_sinks();

        for (auto& sink : sinks) {
            model->remove_sink(sink);
        }
    }

    {
        for (auto& result : results_to_remove) {
            model->remove_result(result);
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

    for (auto& parameter : parameters_to_remove) {
        model->remove_parameter(parameter);
    }

    if (m_use_per_layer_block_indices_inputs) {
        model->add_parameters(block_indices_inputs_for_each_layer);
    }

    if (m_use_score_outputs) {
        model->add_results(score_results);
    }

    if (m_allow_cache_rotation) {
        model->add_parameters(rotated_block_indices_inputs_for_each_layer);
        model->add_parameters(rotation_deltas_inputs_for_each_layer);
        model->add_parameters({model_rotation_trig_lut});
    }

    model->add_parameters(kv_parameters);
    model->add_parameters(model_remaining_params);
    model->add_parameters({std::move(max_context_len)});
    model->validate_nodes_and_infer_types();
    return true;
}
