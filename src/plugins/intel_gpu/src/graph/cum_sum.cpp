// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cum_sum_inst.h"

#include "primitive_type_base.h"
#include "json_object.h"
#include <string>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(cum_sum)

layout cum_sum_inst::calc_output_layout(cum_sum_node const& node, kernel_impl_params const& impl_param) {
    return impl_param.get_input_layout();
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
