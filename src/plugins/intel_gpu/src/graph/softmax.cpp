// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "softmax_inst.h"
#include "primitive_type_base.h"
#include "json_object.h"
#include <string>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(softmax)

std::string softmax_inst::to_string(softmax_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;

    json_composite softmax_info;
    softmax_info.add("dimension", desc->dimension);

    node_info->add("softmax_info", softmax_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

softmax_inst::typed_primitive_inst(network& network, softmax_node const& node) : parent(network, node) {}
}  // namespace cldnn
