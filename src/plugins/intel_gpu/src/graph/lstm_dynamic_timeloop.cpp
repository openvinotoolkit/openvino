// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "lstm_dynamic_timeloop_inst.h"
#include "lstm_dynamic_inst.h"
#include "primitive_type_base.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include <string>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(lstm_dynamic_timeloop)

program_node& lstm_dynamic_timeloop_node::get_dependency_by_name(std::string val) const {
    return get_dependency(get_dependency_idx(val));
}

void lstm_dynamic_timeloop_node::init_params_list() {
    _param_list.push_back("input");
    _param_list.push_back("dyn_length");
    _param_list.push_back("recurrent");
    if (last_hidden_output_term())
        _param_list.push_back("last_hidden_output");
    if (last_cell_output_term())
        _param_list.push_back("last_cell_output");
    if (initial_hidden_term())
        _param_list.push_back("initial_hidden");
    if (initial_cell_term())
        _param_list.push_back("initial_cell");
}

void lstm_dynamic_timeloop_node::reverse_optional_outputs_connections() {
    auto reverse_connections = [&](program_node& mutable_data_node, const std::string& dependency_tag) {
        auto index_to_insert = get_param_list_index(dependency_tag);
        mutable_data_node.dependencies.erase(std::remove_if(mutable_data_node.dependencies.begin(), mutable_data_node.dependencies.end(),
        [&](const std::pair<program_node*, int>& dep) {
            return this == dep.first;
        }));
        mutable_data_node.users.push_back(this);
        users.remove(&mutable_data_node);
        dependencies.insert(dependencies.begin() + index_to_insert, {&mutable_data_node, 0});
        // fix inputs/outputs
        if (mutable_data_node.get_dependencies().empty()) {
            myprog.get_inputs().push_back(&mutable_data_node);
        }
        if (mutable_data_node.is_output()) {
            mutable_data_node.set_output(false);
            auto& program_output = myprog.get_outputs();
            program_output.erase(std::remove(program_output.begin(), program_output.end(), &mutable_data_node));
        }
    };

    if (last_hidden_output_term()) {
        reverse_connections(myprog.get_node(get_primitive()->last_hidden_state), "last_hidden_output");
    }
    if (last_cell_output_term()) {
        reverse_connections(myprog.get_node(get_primitive()->last_cell_state), "last_cell_output");
    }

    // moved mutable data do deps, try to set this node at output if no users
    auto& outputs = myprog.get_outputs();
    if (users.empty() && std::find(outputs.begin(), outputs.end(), this) == outputs.end()) {
        output = true;
        myprog.get_outputs().push_back(this);
    }
}

size_t lstm_dynamic_timeloop_node::get_dependency_idx(std::string val) const {
    auto ret = get_param_list_index(val);
    CLDNN_ERROR_EQUAL(id(),
                      "Dependency index",
                      ret,
                      "out of range number",
                      _param_list.size(),
                      "Trying to get non-exsisting param!");
    return ret;
}

// input_tensor:   [b: batch, f: max_sequence_length, x: 4 * hiden_size, y: direction]
// recurr_tensor:  [b: 1, f: direction, x: hidden_size, y: 4 * hidden_size]
// init_cell:      [b: batch, f: 1, x: hidden_size, y: direction]
// output_tensor:  [b: batch, f: max_sequence_length, x: hidden_size, y: direction]
layout lstm_dynamic_timeloop_inst::calc_output_layout(lstm_dynamic_timeloop_node const& node, kernel_impl_params const& impl_param) {
    assert(static_cast<bool>(impl_param.desc->output_data_types[0]) == false &&
           "Output data type forcing is not supported for lstm_dynamic_node!");
    auto input_layout = impl_param.get_input_layout();
    auto batch = input_layout.batch();
    auto output_sequence = input_layout.feature();
    auto reccurent_layout = node.recurrent().get_output_layout();
    auto hidden_size = reccurent_layout.spatial(0);
    auto direction = reccurent_layout.feature();
    return layout(input_layout.data_type, input_layout.format, tensor(batch, output_sequence, hidden_size, direction));
}

std::string lstm_dynamic_timeloop_inst::to_string(lstm_dynamic_timeloop_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto initial_hidden_id = desc->initial_hidden != "" ? desc->initial_hidden : "no initial hidden";
    auto initial_cell_id = desc->initial_cell != "" ? desc->initial_cell : "no inital cell";
    auto last_cell_id = desc->last_cell_state != "" ? desc->last_cell_state : "no inital cell";
    auto last_hidden_id = desc->last_hidden_state != "" ? desc->last_hidden_state : "no inital hidden";

    std::stringstream primitive_description;
    json_composite lstm_dynamic_input_info;
    lstm_dynamic_input_info.add("dyn_length id", desc->dyn_length);
    lstm_dynamic_input_info.add("recurrent id", desc->recurrent);
    lstm_dynamic_input_info.add("initial cell id", std::move(initial_cell_id));
    lstm_dynamic_input_info.add("initial hidden id", initial_hidden_id);
    lstm_dynamic_input_info.add("last cell id", last_cell_id);
    lstm_dynamic_input_info.add("last hidden id", std::move(last_hidden_id));
    lstm_dynamic_input_info.add("max seq len", node.input().get_output_layout().feature());
    lstm_dynamic_input_info.add("hidden size", node.recurrent().get_output_layout().spatial(0));
    lstm_dynamic_input_info.add("direction", node.recurrent().get_output_layout().feature());
    node_info->add("lstm_dynamic_timeloop info", lstm_dynamic_input_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

lstm_dynamic_timeloop_inst::typed_primitive_inst(network& network, lstm_dynamic_timeloop_node const& node)
    : parent(network, node) {
    auto batch_size = node.get_output_layout().batch();
    auto direction = node.direction();

    // TODO: check input sizes
    auto input_id = node.input().id();
    auto input_layout = node.input().get_output_layout();
    auto hidden_size = input_layout.spatial(0) / 4;
    CLDNN_ERROR_NOT_PROPER_FORMAT(node.id(),
                                  "input format",
                                  input_layout.format.value,
                                  "expected format",
                                  format::bfyx);
    lstm_dynamic_inst::check_direction(node.input(), direction, "input");

    // check recurrent
    CLDNN_ERROR_BOOL(node.id(), "Recurrent memory", !node.recurrent_term(), "Id of weights memory is not set.");
    auto reccurent_id = node.recurrent().id();
    auto recurrent_layout = node.recurrent().get_output_layout();
    CLDNN_ERROR_NOT_PROPER_FORMAT(node.id(),
                                  "recurrent format",
                                  node.recurrent().get_output_layout().format.value,
                                  "expected bfyx format",
                                  format::bfyx);
    CLDNN_ERROR_NOT_EQUAL(node.id(),
                          "Recurrent batch size",
                          recurrent_layout.batch(),
                          "1",
                          1,
                          "Sizes mismatch, reccuren_id: " + reccurent_id);
    if (recurrent_layout.feature() != direction)
        CLDNN_ERROR_MESSAGE(node.id(), "Reccurent directions size needs to be equal to 1 or 2 (bidrectional) !");
    CLDNN_ERROR_NOT_EQUAL(node.id(),
                          "Recurrent x size",
                          recurrent_layout.spatial(0),
                          "hidden_size",
                          hidden_size,
                          "Sizes mismatch, reccuren_id: " + reccurent_id);
    CLDNN_ERROR_NOT_EQUAL(node.id(),
                          "Recurrent y size",
                          recurrent_layout.spatial(1),
                          "4 * hidden_size",
                          4 * hidden_size,
                          "Sizes mismatch, reccuren_id: " + reccurent_id);

    if (initial_cell_term()) {
        lstm_dynamic_inst::check_common_lstm_dynamic_sizes(node.initial_cell(),
                                                           batch_size,
                                                           hidden_size,
                                                           direction,
                                                           "initial_cell");
    }

    if (initial_hidden_term()) {
        lstm_dynamic_inst::check_common_lstm_dynamic_sizes(node.initial_hidden(),
                                                           batch_size,
                                                           hidden_size,
                                                           direction,
                                                           "initial_hidden");
    }

    if (node.last_hidden_output_term()) {
        lstm_dynamic_inst::check_common_lstm_dynamic_sizes(node.last_hidden_state(),
                                                           batch_size,
                                                           hidden_size,
                                                           direction,
                                                           "optional_hidden_output");
    }

    if (node.last_cell_output_term()) {
        lstm_dynamic_inst::check_common_lstm_dynamic_sizes(node.last_cell_state(),
                                                           batch_size,
                                                           hidden_size,
                                                           direction,
                                                           "optional_cell_output");
    }
}
}  // namespace cldnn
