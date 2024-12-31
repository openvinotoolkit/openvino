// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/reverse.hpp"
#include <string>

#include "json_object.h"
#include "primitive_type_base.h"
#include "reverse_inst.h"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(reverse)

std::string reverse_inst::to_string(reverse_node const& node) {
    const auto prim = node.get_primitive();

    std::stringstream primitive_description;

    json_composite info;
    info.add("input id", node.input(0).id());
    info.add("axes id", node.input(1).id());
    const auto mode = prim->mode == ov::op::v1::Reverse::Mode::INDEX ? "index" : "mask";
    info.add("mode", mode);

    auto node_info = node.desc_to_json();
    node_info->add("reverse_info", info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

reverse_inst::typed_primitive_inst(network& network, reverse_node const& node) : parent(network, node) {}

}  // namespace cldnn
