/*
// Copyright (c) 2018 Intel Corporation
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
///////////////////////////////////////////////////////////////////////////////////////////////////
#include "embed_inst.h"
#include "primitive_type_base.h"
#include "error_handler.h"
#include "json_object.h"
#include <string>

namespace cldnn {
primitive_type_id embed_type_id() {
    static primitive_type_base<embed> instance;
    return &instance;
}

layout embed_inst::calc_output_layout(embed_node const& node) {
    assert(static_cast<bool>(node.get_primitive()->output_data_type) == false &&
           "Output data type forcing is not supported for embed_node!");
    auto input_layout = node.input().get_output_layout();
    auto desc = node.get_primitive();
    auto weights_layout = node.weights().get_output_layout();

    auto result =
        layout(input_layout.data_type,
               format::bfyx,
               tensor(input_layout.size.batch[0], input_layout.size.spatial[0], weights_layout.size.batch[0], 1));
    return result;
}

std::string embed_inst::to_string(embed_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto bias_id = desc->bias != "" ? desc->bias : "no bias";
    auto weights_id = desc->weights;

    std::stringstream primitive_description;

    json_composite embed_info;
    embed_info.add("weights id", weights_id);
    embed_info.add("bias id", bias_id);

    node_info->add("embed info", embed_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

embed_inst::typed_primitive_inst(network_impl& network, embed_node const& node) : parent(network, node) {
    auto input_size = node.input().get_output_layout();
    auto output_size = output_memory().get_layout();
    CLDNN_ERROR_NOT_PROPER_FORMAT(node.id(),
                                  "input format",
                                  input_size.format.value,
                                  "expected format",
                                  format::yxfb,
                                  format::bfyx);
    CLDNN_ERROR_NOT_EQUAL(node.id(),
                          "Input size",
                          input_size.size.raw.size(),
                          "output size",
                          output_size.size.raw.size(),
                          "");
    CLDNN_ERROR_NOT_EQUAL(node.id(),
                          "Input batch",
                          input_size.size.batch[0],
                          "output batch",
                          output_size.size.batch[0],
                          "");
    CLDNN_ERROR_NOT_EQUAL(node.id(), "Input feature", input_size.size.feature[0], "size 1", 1, "");
    CLDNN_ERROR_NOT_EQUAL(node.id(), "Input y size ", input_size.size.spatial[1], "size 1", 1, "");
}
}  // namespace cldnn
