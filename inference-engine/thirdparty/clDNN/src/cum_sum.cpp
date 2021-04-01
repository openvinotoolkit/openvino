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

#include "cum_sum_inst.h"

#include "primitive_type_base.h"
#include "error_handler.h"
#include "json_object.h"
#include <string>

namespace cldnn {
primitive_type_id cum_sum::type_id() {
    static primitive_type_base<cum_sum> instance;
    return &instance;
}

layout cum_sum_inst::calc_output_layout(cum_sum_node const& node) {
    return node.input(0).get_output_layout();
}

std::string cum_sum_inst::to_string(cum_sum_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto& input = node.input();

    std::stringstream primitive_description;

    json_composite cum_sum_info;
    cum_sum_info.add("input id", input.id());
    cum_sum_info.add("exclusive", desc->exclusive);
    cum_sum_info.add("reverse", desc->reverse);

    node_info->add("cum_sum info", cum_sum_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

cum_sum_inst::typed_primitive_inst(network_impl& network, cum_sum_node const& node)
    : parent(network, node) {}

}  // namespace cldnn
