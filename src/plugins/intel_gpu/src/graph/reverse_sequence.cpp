// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reverse_sequence_inst.h"

#include "primitive_type_base.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include <string>

namespace cldnn {
primitive_type_id reverse_sequence::type_id() {
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

reverse_sequence_inst::typed_primitive_inst(network& network, reverse_sequence_node const& node)
    : parent(network, node) {}

}  // namespace cldnn
