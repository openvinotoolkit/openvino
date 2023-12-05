// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "lstm_inst.h"
#include "primitive_type_base.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include <string>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(lstm)

layout lstm_inst::calc_output_layout(lstm_node const& node, kernel_impl_params const& impl_param) {
    assert(static_cast<bool>(impl_param.desc->output_data_types[0]) == false &&
           "Output data type forcing is not supported for lstm_node!");
    auto input_layout = impl_param.get_input_layout();
    auto hidden_layout = node.inital_hidden().get_output_layout();

    // input     = [ batch,  sequence,       direction,      input_size ]
    // weights   = [     1, direction, 4 * hidden_size,      input_size ]
    // recurrent = [     1, direction, 4 * hidden_size,     hidden_size ]
    // biases    = [     1,         1,       direction, 4 * hidden_size ]
    // hidden    = [ batch,         1,       direction,     hidden_size ]
    // cell      = [ batch,         1,       direction,     hidden_size ]
    // output    = [ batch,  sequence,       direction,     hidden_size ]
    auto result = layout(input_layout.data_type,
                         format::bfyx,
                         tensor(hidden_layout.feature(),
                                input_layout.feature(),
                                hidden_layout.spatial(0),
                                hidden_layout.spatial(1)));
    return result;
}

std::string lstm_inst::to_string(lstm_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto weights_id = desc->weights;
    auto recurrent_id = desc->recurrent;
    auto bias_id = desc->bias != "" ? desc->bias : "no bias";
    auto peepholes_id = desc->peepholes != "" ? desc->peepholes : "no peepholes";
    auto initial_hidden_id = desc->initial_hidden != "" ? desc->initial_hidden : "no inital hidden";
    auto initial_cell_id = desc->initial_cell != "" ? desc->initial_cell : "no initial cell";

    std::stringstream primitive_description;

    json_composite lstm_info;
    lstm_info.add("weights id", weights_id);
    lstm_info.add("recurrent id", recurrent_id);
    lstm_info.add("bias id", std::move(bias_id));
    lstm_info.add("peepholes id", std::move(peepholes_id));
    lstm_info.add("initial_hidden id", std::move(initial_hidden_id));
    lstm_info.add("initial_cell id", std::move(initial_cell_id));
    node_info->add("lstm info", lstm_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

lstm_inst::typed_primitive_inst(network& network, lstm_node const& node) : parent(network, node) {
    auto input_layout = node.input().get_output_layout();
    CLDNN_ERROR_NOT_PROPER_FORMAT(node.id(),
                                  "input format",
                                  input_layout.format.value,
                                  "expected format",
                                  format::bfyx);
}

}  // namespace cldnn
