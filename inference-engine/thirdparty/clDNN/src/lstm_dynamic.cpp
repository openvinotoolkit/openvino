/*
// Copyright (c) 2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "lstm_dynamic_inst.h"
#include "primitive_type_base.h"
#include "error_handler.h"
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

lstm_dynamic_inst::typed_primitive_inst(network_impl& network, lstm_dynamic_node const& node) : parent(network, node) {
    CLDNN_ERROR_MESSAGE(node.id(),
                        std::string("This primitive_inst should never be created. It should be repalced by ")
                        .append("lstm_dynamic_input + lstm_dyamic_timeloop combinations."));
}

}  // namespace cldnn
