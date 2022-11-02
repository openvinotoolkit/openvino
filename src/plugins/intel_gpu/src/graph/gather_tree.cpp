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

layout gather_tree_inst::calc_output_layout(gather_tree_node const& node, kernel_impl_params const& impl_param) {
    assert(static_cast<bool>(impl_param.desc->output_data_type) == false &&
        "Output data type forcing is not supported for gather_tree_node!");
    auto input_layout = impl_param.get_input_layout();
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
        format::byxf,
        format::b_fs_yx_fsv16,
        format::b_fs_yx_fsv32,
        format::bs_fs_yx_bsv4_fsv4,
        format::bs_fs_yx_bsv8_fsv4,
        format::bs_fs_yx_bsv8_fsv2,
        format::bs_fs_yx_bsv4_fsv2,
        format::bs_fs_yx_bsv16_fsv16,
        format::bs_fs_yx_bsv32_fsv16,
        format::bs_fs_yx_bsv32_fsv32);

    auto dependencies = node.get_dependencies();

    // check input dims
    CLDNN_ERROR_NOT_EQUAL(node.id(),
        "input0 size", dependencies.at(0)->get_output_layout().get_tensor(), "output size", input_layout.get_tensor(),
        "mismatch");
    CLDNN_ERROR_NOT_EQUAL(node.id(),
        "input1 size", dependencies.at(1)->get_output_layout().get_tensor(), "output size", input_layout.get_tensor(),
        "mismatch");
    CLDNN_ERROR_NOT_EQUAL(node.id(),
        "input2 size", dependencies.at(2)->get_output_layout().count(), "node's feature size", input_layout.feature(),
        "There can't be more than one end_token");
    CLDNN_ERROR_NOT_EQUAL(node.id(),
        "input3 size", dependencies.at(3)->get_output_layout().count(), "one", 1,
        "There can't be more than one end_token");
}
}  // namespace cldnn
