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

#include "reverse_sequence_inst.h"

#include "primitive_type_base.h"
#include "error_handler.h"
#include "json_object.h"
#include <string>

namespace cldnn {
primitive_type_id reverse_sequence_type_id() {
    static primitive_type_base<reverse_sequence> instance;
    return &instance;
}

layout reverse_sequence_inst::calc_output_layout(reverse_sequence_node const& node) {
    auto desc = node.get_primitive();

    auto input_layout = node.input(0).get_output_layout();
    auto input_format = input_layout.format;

    return layout{input_layout.data_type, input_format, input_layout.size};
}

std::string reverse_sequence_inst::to_string(reverse_sequence_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;

    json_composite reverse_sequence_info;
    reverse_sequence_info.add("input id", node.input(0).id());
    reverse_sequence_info.add("sequence lengths id", node.input(1).id());
    reverse_sequence_info.add("sequence axis", desc->seq_axis);
    reverse_sequence_info.add("batch axis", desc->batch_axis);

    node_info->add("reverse_sequence info", reverse_sequence_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

reverse_sequence_inst::typed_primitive_inst(network_impl& network, reverse_sequence_node const& node)
    : parent(network, node) {}

}  // namespace cldnn
