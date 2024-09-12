// Copyright (C) 2018-2024 Intel Corporation
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

ov::pass::SDPAToPagedAttention::SDPAToPagedAttention(bool use_block_indices_inputs, bool use_score_outputs)
    : m_use_block_indices_inputs(use_block_indices_inputs),
      m_use_score_outputs(use_score_outputs) {}

static std::shared_ptr<v0::Parameter> setName(std::shared_ptr<v0::Parameter> node, const char* name) {
    // Set name for both node and output tensor (should be only one tensor, and any other names will be overriden by a
    // given single name)
    node->set_friendly_name(name);
    OPENVINO_ASSERT(node->get_output_size() == 1);  // Should I use assert here?
    node->get_output_tensor(0).set_names({name});
    return node;
}

bool ov::pass::SDPAToPagedAttention::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_MODEL_SCOPE(SDPAToPagedAttention);

    OPENVINO_ASSERT(ov::op::util::has_op_with_type<ov::op::v13::ScaledDotProductAttention>(model),
                    "No ScaledDotProductAttention operation observed in the graph, cannot perform"
                    "the SDPAToPagedAttention transformation.");

    auto max_context_len = setName(std::make_shared<v0::Parameter>(element::i32, PartialShape{}), "max_context_len");
    ParameterVector model_remaining_params{
        setName(std::make_shared<v0::Parameter>(element::i32, PartialShape{-1}), "past_lens"),
        setName(std::make_shared<v0::Parameter>(element::i32, PartialShape{-1}), "subsequence_begins"),
        setName(std::make_shared<v0::Parameter>(element::i32, PartialShape{-1}), "block_indices_begins"),
    };
    if (!m_use_block_indices_inputs) {
        auto block_indices = setName(std::make_shared<v0::Parameter>(element::i32, PartialShape{-1}), "block_indices");
        model_remaining_params.insert(model_remaining_params.begin() + 2, block_indices);
    }

    auto sliding_window = v0::Constant::create(element::i32, Shape{}, {0});  // sliding_window

    std::shared_ptr<v0::Parameter> input_ids_node =
        std::dynamic_pointer_cast<v0::Parameter>(model->input("input_ids").get_node_shared_ptr());
    input_ids_node->set_partial_shape(PartialShape{-1});
    auto unsqueezed_input_ids =
        std::make_shared<v0::Unsqueeze>(input_ids_node, v0::Constant::create(element::i32, Shape{}, {1}));
    replace_node(input_ids_node, unsqueezed_input_ids);

    auto cur_seq_len = std::make_shared<v8::Gather>(std::make_shared<v3::ShapeOf>(unsqueezed_input_ids),
                                                    v0::Constant::create(element::i64, Shape{}, {1}),
                                                    v0::Constant::create(element::i64, Shape{}, {0}));
    auto prev_max_seq_len =
        std::make_shared<v1::Subtract>(max_context_len, std::make_shared<v0::Convert>(cur_seq_len, element::i32));

    auto has_parameter = [=](const std::shared_ptr<ov::Model>& model, const std::string& name) -> bool {
        for (auto& t : model->inputs()) {
            const auto& names = t.get_names();
            if (names.find(name) != names.end()) {
                return true;
            }
        }

        return false;
    };

    ParameterVector kv_parameters;
    ParameterVector parameters_to_remove;
    ResultVector results_to_remove;  // # used, but cannot really track all Results in stateless model
    ParameterVector block_indices_inputs;
    ResultVector score_results;

    std::shared_ptr<v0::Parameter> position_ids;
    if (!has_parameter(model, "position_ids")) {
        position_ids = setName(std::make_shared<v0::Parameter>(element::i64, PartialShape{-1}), "position_ids");
        model->add_parameters({position_ids});
    } else {
        position_ids = std::dynamic_pointer_cast<v0::Parameter>(model->input("position_ids").get_node_shared_ptr());
        position_ids->set_partial_shape(PartialShape{-1});
        position_ids->validate_and_infer_types();
    }
    auto unsqueezed_position_ids =
        std::make_shared<v0::Unsqueeze>(position_ids, v0::Constant::create(element::i32, Shape{}, {1}));
    replace_node(position_ids, unsqueezed_position_ids);

    int layer_index = 0;

    auto batch_dim =
        std::make_shared<v3::ShapeOf>(position_ids);  // it is not always required, so will be disposed if not needed

    ov::pass::Manager manager("SDPA to PA");
    manager.set_per_pass_validation(false);
    manager.register_pass<StateManagementPattern>(kv_parameters,
                                                  model_remaining_params,
                                                  sliding_window,
                                                  parameters_to_remove,
                                                  layer_index,
                                                  max_context_len->output(0),
                                                  block_indices_inputs,
                                                  score_results,
                                                  m_use_block_indices_inputs,
                                                  m_use_score_outputs);
    manager.register_pass<PrevSequenceLengthPattern>(prev_max_seq_len, batch_dim);
    manager.register_pass<TotalSequenceLengthPattern>(max_context_len);
    manager.register_pass<PositionIDsReplacer>(unsqueezed_position_ids->output(0));
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
        if (has_parameter(model, param_name)) {
            if (const auto& param =
                    std::dynamic_pointer_cast<v0::Parameter>(model->input(param_name).get_node_shared_ptr())) {
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
            } else {
                OPENVINO_THROW("The model is in the inconsistent state. Found input '",
                               param_name,
                               "', but couldn't cast it to v0::Parameter.");
                return false;
            }
        }
    }

    for (auto& parameter : parameters_to_remove) {
        model->remove_parameter(parameter);
    }

    if (m_use_block_indices_inputs) {
        model->add_parameters(block_indices_inputs);
    }

    if (m_use_score_outputs) {
        model->add_results(score_results);
    }

    model->add_parameters(kv_parameters);
    model->add_parameters(model_remaining_params);
    model->add_parameters({std::move(max_context_len)});
    model->validate_nodes_and_infer_types();
    return true;
}
