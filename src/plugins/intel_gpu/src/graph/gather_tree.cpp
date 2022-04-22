// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gather_tree_inst.h"

#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include "primitive_type_base.h"
#include <string>
#include <algorithm>

namespace cldnn {
primitive_type_id gather_tree::type_id() {
    static primitive_type_base<gather_tree> instance;
    return &instance;
}

layout gather_tree_inst::calc_output_layout(gather_tree_node const& node) {
    assert(static_cast<bool>(node.get_primitive()->output_data_types.at(0)) == false &&
        "Output data type forcing is not supported for gather_tree_node!");
    auto input_layout = node.input().get_output_layout();
    return input_layout;
}

std::string gather_tree_inst::to_string(gather_tree_node const& node) {
    std::stringstream primitive_description;
    node.desc_to_json()->dump(primitive_description);
    return primitive_description.str();
}

gather_tree_inst::typed_primitive_inst(network& network, gather_tree_node const& node) : parent(network, node) {
    auto input_layout = node.input().get_output_layout();

    const auto input_format = input_layout.format;

    CLDNN_ERROR_NOT_PROPER_FORMAT(node.id(),
        "Input format",
        input_format.value,
        "supported border primitive input formats",
        format::bfyx,
        format::yxfb,
        format::byxf);

    auto dependencies = node.get_dependencies();

    // check input dims
    CLDNN_ERROR_NOT_EQUAL(node.id(),
        "input0 size", dependencies.at(0).first->get_output_layout().size, "output size", input_layout.size,
        "mismatch");
    CLDNN_ERROR_NOT_EQUAL(node.id(),
        "input1 size", dependencies.at(1).first->get_output_layout().size, "output size", input_layout.size,
        "mismatch");
    CLDNN_ERROR_NOT_EQUAL(node.id(),
        "input2 size", dependencies.at(2).first->get_output_layout().count(), "node's feature size", input_layout.size.feature.at(0),
        "There can't be more than one end_token");
    CLDNN_ERROR_NOT_EQUAL(node.id(),
        "input3 size", dependencies.at(3).first->get_output_layout().size.count(), "one", 1,
        "There can't be more than one end_token");
}
}  // namespace cldnn
