// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "generic_layer_inst.h"
#include "primitive_type_base.h"

#include "json_object.h"

#include <algorithm>
#include <string>
#include <memory>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(generic_layer)

generic_layer_node::typed_program_node(const std::shared_ptr<generic_layer> prim, program& prog)
    : parent(prim, prog) {
    can_share_buffer(false);
}

generic_layer_inst::typed_primitive_inst(network& network, generic_layer_node const& node)
    : parent(network, node) {}

std::string generic_layer_inst::to_string(generic_layer_node const& node) {
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;

    node_info->dump(primitive_description);

    return primitive_description.str();
}

}  // namespace cldnn
