/*
// Copyright (c) 2016 Intel Corporation
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
#include "lstm_inst.h"
#include "primitive_type_base.h"
#include "error_handler.h"
#include "json_object.h"

namespace cldnn
{
primitive_type_id lstm_type_id()
{
    static primitive_type_base<lstm> instance;
    return &instance;
}


layout lstm_inst::calc_output_layout(lstm_node const& node)
{
    auto input_layout = node.input().get_output_layout();
    auto hidden_layout = node.inital_hidden().get_output_layout();

    // input     = [        1,  sequence,           batch,      input_size ]
    // weights   = [        1, direction, 4 * hidden_size,      input_size ]
    // recurrent = [        1, direction, 4 * hidden_size,     hidden_size ]
    // biases    = [        1,         1,       direction, 4 * hidden_size ]
    // hidden    = [        1, direction,           batch,     hidden_size ]
    // cell      = [        1, direction,           batch,     hidden_size ]
    // output    = [ sequence, direction,           batch,     hidden_size ]    
	auto result = layout(input_layout.data_type, format::bfyx,
                  tensor(hidden_layout.size.feature[0], input_layout.size.feature[0], hidden_layout.size.spatial[0], hidden_layout.size.spatial[1]));
    return result;
}

std::string lstm_inst::to_string(lstm_node const& node)
{
    auto desc         = node.get_primitive();
    auto node_info    = node.desc_to_json();
    auto weights_id   = desc->weights;
    auto recurrent_id = desc->recurrent;
    auto bias_id      = desc->bias != "" ? desc->bias : "no bias";
    auto peepholes_id = desc->peepholes != "" ? desc->peepholes : "no peepholes";
    auto initial_hidden_id = desc->initial_hidden != "" ? desc->initial_hidden : "no inital hidden";
    auto initial_cell_id = desc->initial_cell != "" ? desc->initial_cell : "no initial cell";

    std::stringstream primitive_description;

    json_composite lstm_info;
    lstm_info.add("weights id", weights_id);
    lstm_info.add("recurrent id", recurrent_id);
    lstm_info.add("bias id", bias_id);
    lstm_info.add("peepholes id", peepholes_id);
    lstm_info.add("initial_hidden id", initial_hidden_id);
    lstm_info.add("initial_cell id", initial_cell_id);
    node_info->add("lstm info", lstm_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

lstm_inst::typed_primitive_inst(network_impl& network, lstm_node const& node)
    :parent(network, node)
{
    // [ARIEL] TODO: That do we need to check here??
    auto input_size = node.input().get_output_layout();
    // auto output_size = output_memory().get_layout();
    CLDNN_ERROR_NOT_PROPER_FORMAT(node.id(), "input format", input_size.format.value, "expected format", format::bfyx);
    //CLDNN_ERROR_NOT_EQUAL(node.id(), "Input size", input_size.size.raw.size(), "output size", output_size.size.raw.size(), "");
}
}
