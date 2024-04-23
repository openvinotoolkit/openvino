// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/sdpa_to_paged_attention.hpp"

#include "openvino/cc/pass/itt.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "transformations/utils/utils.hpp"
#include "openvino/pass/manager.hpp"

using namespace ov::op;

bool ov::pass::SDPAToPagedAttention::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_MODEL_SCOPE(SDPAToPagedAttention);

    // писать сюда
    // yes sir
    std::cout << "[ STUB ] ov::pass::SDPAToPagedAttention::run_on_model\n";

    //Should I use PartialShape or Shape??
    //Do I need to specify name because in Python it looks like the name argument is ignored completely
    // Maybe I should specify it explicitly
    auto max_context_len = std::make_shared<v0::Parameter>(element::i64, PartialShape{}); // max_context_len
    ParameterVector model_remaining_param = {
        std::make_shared<v0::Parameter>(element::boolean, PartialShape{}), // is_prompt
        std::make_shared<v0::Parameter>(element::i64, PartialShape{-1, -1}),
        max_context_len,
        std::make_shared<v0::Parameter>(element::i64, PartialShape{-1}), // context_lens
        std::make_shared<v0::Parameter>(element::i64, PartialShape{-1, -1}), // block_tables
    };

    for (auto& parameter : model_remaining_param) {
        parameter->get_output_tensor(0).set_names({parameter->get_friendly_name()});
    }
    // Is it a right way to create a sliding window?
    auto sliding_window = v0::Constant::create(element::i32, Shape{}, {0}); // sliding_window

    auto cur_seq_len = std::make_shared<v1::Gather>(v3::ShapeOf(model->input("input_ids")).output(0),
                                                    v0::Constant::create(element::i32, Shape{}, {1}),
                                                    v0::Constant::create(element::i32, Shape{}, {0}));
    cur_seq_len->set_friendly_name("my_curr_seq_len");

    auto prev_max_seq_len = std::make_shared<v1::Subtract>(max_context_len, cur_seq_len);

    auto has_parameter = [OV_CAPTURE_CPY_AND_THIS](const std::shared_ptr<ov::Model>& model, const std::string& name) -> bool {
        for (auto& t : model->inputs()) {
            const auto& names = t.get_names();
            if (names.find(name) != names.end()) {
                return true;
            }
        }

        return false;
    };

    ParameterVector kv_parameters;
    std::vector<std::shared_ptr<Node>> assignes_to_remove; // not really used
    ParameterVector parameters_to_remove;
    ResultVector results_to_remove;

    if (!has_parameter(model, "position_id")) {
        auto position_ids = std::make_shared<v0::Parameter>(element::i64, PartialShape{-1, -1}); // position_ids SHOULD I SET NAME?
        position_ids->get_output_tensor(0).set_names({position_ids->get_friendly_name()});
        model->add_parameters({position_ids});
        std::cout << "CREATED A NEW position_ids PARAMETER" << std::endl;
    }

    auto position_ids_input = model->input("position_id"); // Why is it needed?


    ov::pass::Manager manager;
    manager.set_per_pass_validation(false);
    // manager.register_pass();
    // manager.register_pass();
    // manager.register_pass();

    // manager.register_pass();

    manager.run_passes(model);

    if (has_parameter(model, "beam_idx")) {
        if (const auto& parameter = std::dynamic_pointer_cast<v0::Parameter>(model->input("beam_idx").get_node_shared_ptr())) {
            model->remove_parameter(parameter);
        } else {
            return false;
        }
    }

    if (const auto& parameter = std::dynamic_pointer_cast<v0::Parameter>(model->input("attention_mask").get_node_shared_ptr())) {
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
    model->add_parameters(model_remaining_param);

    return true;
}
