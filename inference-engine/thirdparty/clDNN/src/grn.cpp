/*
// Copyright (c) 2020 Intel Corporation
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

#include "grn_inst.h"
#include "primitive_type_base.h"
#include "json_object.h"
#include <string>

namespace cldnn {
primitive_type_id grn::type_id() {
    static primitive_type_base<grn> instance;
    return &instance;
}

layout grn_inst::calc_output_layout(grn_node const& node) {
    auto input_node_layout = node.input().get_non_padded_output_layout();
    auto output_type = node.get_primitive()->output_data_type ? *node.get_primitive()->output_data_type : input_node_layout.data_type;

    return layout(output_type, input_node_layout.format, input_node_layout.size);
}

std::string grn_inst::to_string(grn_node const& node) {
    auto node_info = node.desc_to_json();
    auto desc = node.get_primitive();
    auto bias = desc->bias;
    auto& input = node.input();

    std::stringstream primitive_description;

    json_composite grn_info;
    grn_info.add("input id", input.id());
    grn_info.add("bias", bias);

    node_info->add("grn info", grn_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

grn_inst::typed_primitive_inst(network_impl& network, grn_node const& node) : parent(network, node) {}
}  // namespace cldnn
