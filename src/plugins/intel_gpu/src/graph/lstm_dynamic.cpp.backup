// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "lstm_dynamic_inst.h"
#include "primitive_type_base.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include <string>

namespace cldnn {
primitive_type_id lstm_dynamic::type_id() {
    static primitive_type_base<lstm_dynamic> instance;
    return &instance;
}

// input_tensor:   [b: batch, f: max_sequence_length, x: input_size, y: direction]
// weights_tensor: [b: 1, f: direction, x: input_size, y: 4 * hidden_size]
// recurr_tensor:  [b: 1, f: direction, x: hidden_size, y: 4 * hidden_size]
// init_hidden:    [b: batch, f: 1, x: hidden_size, y: direction]
// init_cell:      [b: batch, f: 1, x: hidden_size, y: direction]
// output_tensor:  [b: batch, f: max_sequence_length, x: hidden_size, y: direction]
layout lstm_dynamic_inst::calc_output_layout(lstm_dynamic_node const& node) {
    assert(static_cast<bool>(node.get_primitive()->output_data_type) == false &&
           "Output data type forcing is not supported for lstm_dynamic_node!");
    /*
        This program node is just placeholder for input + timeloop combinations, thus this is returning dummy layout.
        */
    return node.get_dependency(0).get_output_layout();
}

std::string lstm_dynamic_inst::to_string(lstm_dynamic_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto weights_id = desc->weights;
    auto recurrent_id = desc->recurrent;
    auto bias_id = desc->bias != "" ? desc->bias : "no bias";
    auto initial_hidden_id = desc->initial_hidden != "" ? desc->initial_hidden : "no inital hidden";
    auto initial_cell_id = desc->initial_cell != "" ? desc->initial_cell : "no initial cell";

    std::stringstream primitive_description;
    json_composite lstm_dynamic_info;
    lstm_dynamic_info.add("dyn_length id", desc->dyn_length);
    lstm_dynamic_info.add("weights id", weights_id);
    lstm_dynamic_info.add("recurrent id", recurrent_id);
    lstm_dynamic_info.add("bias id", bias_id);
    lstm_dynamic_info.add("initial_hidden id", initial_hidden_id);
    lstm_dynamic_info.add("initial_cell id", initial_cell_id);
    node_info->add("lstm_dynamic info", lstm_dynamic_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

lstm_dynamic_inst::typed_primitive_inst(network& network, lstm_dynamic_node const& node) : parent(network, node) {
    CLDNN_ERROR_MESSAGE(node.id(),
                        std::string("This primitive_inst should never be created. It should be repalced by ")
                        .append("lstm_dynamic_input + lstm_dyamic_timeloop combinations."));
}

}  // namespace cldnn
