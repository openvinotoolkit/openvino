// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shuffle_channels_inst.h"

#include "primitive_type_base.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include <string>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(shuffle_channels)

std::string shuffle_channels_inst::to_string(shuffle_channels_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto& input = node.input();

    std::stringstream primitive_description;

    json_composite shuffle_channels_info;
    shuffle_channels_info.add("input id", input.id());
    shuffle_channels_info.add("groups number", desc->group);
    shuffle_channels_info.add("axis", desc->axis);

    node_info->add("shuffle_channels info", shuffle_channels_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

shuffle_channels_inst::typed_primitive_inst(network& network, shuffle_channels_node const& node)
    : parent(network, node) {}

}  // namespace cldnn
