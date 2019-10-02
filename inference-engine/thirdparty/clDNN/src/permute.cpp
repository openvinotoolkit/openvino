/*
// Copyright (c) 2016-2019 Intel Corporation
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
#include "permute_inst.h"
#include "primitive_type_base.h"
#include "error_handler.h"
#include "json_object.h"

#include <algorithm>
#include <string>
#include <vector>

namespace cldnn {

primitive_type_id permute::type_id() {
    static primitive_type_base<permute> instance;
    return &instance;
}

layout permute_inst::calc_output_layout(permute_node const& node) {
    assert(static_cast<bool>(node.get_primitive()->output_data_type) == false &&
           "Output data type forcing is not supported for permute_node!");
    auto input_layout = node.input().get_output_layout();
    auto permute_order = node.get_primitive()->permute_order;
    std::vector<tensor::value_type> output_sizes;

    for (size_t x = 0; x < permute_order.size(); x++) {
        output_sizes.push_back(input_layout.size.raw[permute_order[x]]);
    }

    auto input_size = tensor(output_sizes);
    auto op = node.get_primitive()->output_padding;

    return layout(input_layout.data_type, input_layout.format, input_size, op);
}

std::string permute_inst::to_string(permute_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto permute_order = desc->permute_order;
    auto& input = node.input();

    std::stringstream primitive_description;
    std::stringstream ss_permute_order;

    for (size_t i = 0; i < permute_order.size(); ++i) {
        ss_permute_order << permute_order.at(i);
        i != (permute_order.size() - 1) ? ss_permute_order << ", " : ss_permute_order << "";
    }

    json_composite permute_info;
    permute_info.add("input id", input.id());
    permute_info.add("permute order", ss_permute_order.str());

    node_info->add("permute info", permute_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

permute_inst::typed_primitive_inst(network_impl& network, permute_node const& node) : parent(network, node) {
    auto permute_order = argument.permute_order;

    CLDNN_ERROR_LESS_THAN(node.id(),
                          "Permute order size",
                          permute_order.size(),
                          "minimum order size",
                          4,
                          "Permute order size needs to be at least 4.");

    auto required_order_values_size = static_cast<uint32_t>(permute_order.size());

    for (decltype(required_order_values_size) i = 0; i < required_order_values_size; i++) {
        if (!(std::find(permute_order.begin(), permute_order.end(), i) != permute_order.end()))
            CLDNN_ERROR_MESSAGE(node.id(), "Permute order does not contain all of required values.");
    }
}
}  // namespace cldnn
