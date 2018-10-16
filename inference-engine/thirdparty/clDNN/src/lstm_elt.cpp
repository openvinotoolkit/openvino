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
#include "lstm_elt_inst.h"
#include "primitive_type_base.h"
#include "error_handler.h"
#include "json_object.h"

namespace cldnn
{
primitive_type_id lstm_elt_type_id()
{
    static primitive_type_base<lstm_elt> instance;
    return &instance;
}

layout lstm_elt_inst::calc_output_layout(lstm_elt_node const& node)
{
    auto desc = node.get_primitive();
    auto input_layout = node.input().get_output_layout();

    // tempGEMM{bfyx} = [b: batch, f: direction, x: 1,         y: 4 * hidden_size ] input
    // cell{bfyx}     = [b: batch, f: direction, x: 1,         y: hidden_size ] optional
    // output{bfyx}   = [b: batch, f: 2,         x: direction, y: hidden_size ] output
    // The output of the lstm_elt node is the concatenation of the intermediate [hidden, cell] tensors.
    // A crop/split node is needed to extract each individual tensors
    auto result = layout(input_layout.data_type, format::bfyx,
                    tensor(input_layout.size.batch[0], 2, input_layout.size.spatial[0] / 4, input_layout.size.feature[0]));
    return result;
}

std::string lstm_elt_inst::to_string(lstm_elt_node const& node)
{
    auto desc      = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto cell_id   = desc->cell;

    std::stringstream primitive_description;

    json_composite lstm_elt_info;
    lstm_elt_info.add("cell id", cell_id);
    node_info.add("lstm elt info", lstm_elt_info);
    node_info.dump(primitive_description);

    return primitive_description.str();
}

lstm_elt_inst::typed_primitive_inst(network_impl& network, lstm_elt_node const& node)
    :parent(network, node)
{
    auto input_size = node.input().get_output_layout();
    CLDNN_ERROR_NOT_PROPER_FORMAT(node.id(), "input format", input_size.format.value, "expected format", format::bfyx);
}
}
