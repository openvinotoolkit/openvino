// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cum_sum_inst.h"

#include "primitive_type_base.h"
#include "intel_gpu/runtime/error_handler.hpp"
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

cum_sum_inst::typed_primitive_inst(network& network, cum_sum_node const& node)
    : parent(network, node) {}

}  // namespace cldnn
