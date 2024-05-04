// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/sdpa_to_paged_attention.hpp"

#include "openvino/cc/pass/itt.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/sdpa_to_paged_attention/position_ids_replacer.hpp"
#include "transformations/sdpa_to_paged_attention/prev_sequence_length_pattern.hpp"
#include "transformations/sdpa_to_paged_attention/state_management_pattern.hpp"
#include "transformations/sdpa_to_paged_attention/total_sequence_length_pattern.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov::op;

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
    auto max_context_len =
        setName(std::make_shared<v0::Parameter>(element::i64, PartialShape{}), "max_context_len");  // max_context_len
    ParameterVector model_remaining_params = {
        setName(std::make_shared<v0::Parameter>(element::boolean, PartialShape{}), "is_prompt"),  // is_prompt
        setName(std::make_shared<v0::Parameter>(element::i64, PartialShape{-1, -1}), "slot_mapping"),
        max_context_len,
        setName(std::make_shared<v0::Parameter>(element::i64, PartialShape{-1}), "context_lens"),      // context_lens
        setName(std::make_shared<v0::Parameter>(element::i32, PartialShape{-1, -1}), "block_tables"),  // block_tables
    };
    auto sliding_window = v0::Constant::create(element::i32, Shape{}, {0});  // sliding_window

    auto cur_seq_len = std::make_shared<v1::Gather>(std::make_shared<v3::ShapeOf>(model->input("input_ids")),
                                                    v0::Constant::create(element::i64, Shape{}, {1}),
                                                    v0::Constant::create(element::i64, Shape{}, {0}));
    auto prev_max_seq_len = std::make_shared<v1::Subtract>(max_context_len, cur_seq_len);

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
    std::vector<std::shared_ptr<Node>> assignes_to_remove;  // not really used
    ParameterVector parameters_to_remove;
    ResultVector results_to_remove;  // # used, but cannot really track all Results in stateless model

    if (!has_parameter(model, "position_ids")) {
        auto position_ids =
            setName(std::make_shared<v0::Parameter>(element::i64, PartialShape{-1, -1}), "position_ids");
        model->add_parameters({position_ids});
    }
    auto position_ids = std::make_shared<Output<Node>>(model->input("position_ids"));

    int layer_index = 0;

    ov::pass::Manager manager;
    manager.set_per_pass_validation(false);
    manager.register_pass<StateManagementPattern>(kv_parameters,
                                                  model_remaining_params,
                                                  sliding_window,
                                                  parameters_to_remove,
                                                  assignes_to_remove,
                                                  layer_index);
    manager.register_pass<PrevSequenceLengthPattern>(prev_max_seq_len);
    manager.register_pass<TotalSequenceLengthPattern>(max_context_len);

    manager.register_pass<PositionIDsReplacer>(position_ids);

    manager.run_passes(model);

    if (has_parameter(model, "beam_idx")) {
        if (const auto& parameter =
                std::dynamic_pointer_cast<v0::Parameter>(model->input("beam_idx").get_node_shared_ptr())) {
            model->remove_parameter(parameter);
        } else {
            return false;
        }
    }

    if (const auto& parameter =
            std::dynamic_pointer_cast<v0::Parameter>(model->input("attention_mask").get_node_shared_ptr())) {
        model->remove_parameter(parameter);
    } else {
        return false;
    }

    for (auto& parameter : parameters_to_remove) {
        model->remove_parameter(parameter);
    }
    // Remove all Assigns aggressively, the path from the kv-cache concat to Assign can be complicated,
    // but there is no reason to track it and reject part of the Assigns, because the model will remain
    // in incorrect form anyway.
    auto sinks = model->get_sinks();

    for (auto& sink : sinks) {
        model->remove_sink(sink);
    }

    for (auto& result : results_to_remove) {
        model->remove_result(result);
    }

    model->add_parameters(kv_parameters);
    model->add_parameters(model_remaining_params);
    return true;
}
