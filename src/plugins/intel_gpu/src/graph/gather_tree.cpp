// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gather_tree_inst.h"
#include "gather_tree_shape_inference.hpp"

#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include "primitive_type_base.h"
#include <string>
#include <algorithm>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(gather_tree)

layout gather_tree_inst::calc_output_layout(gather_tree_node const& node, kernel_impl_params const& impl_param) {
    assert(static_cast<bool>(impl_param.desc->output_data_types[0]) == false &&
        "Output data type forcing is not supported for gather_tree_node!");
    auto input_layout = impl_param.get_input_layout();
    return input_layout;
}

template<typename ShapeType>
std::vector<layout> gather_tree_inst::calc_output_layouts(gather_tree_node const& /*node*/, const kernel_impl_params& impl_param) {
    auto desc = impl_param.typed_desc<gather_tree>();
    auto input0_layout = impl_param.get_input_layout(0);

    auto output_type = input0_layout.data_type;
    if (impl_param.has_fused_primitives()) {
        output_type = impl_param.get_output_element_type();
    }

    ov::op::v1::GatherTree op;

    std::vector<ShapeType> input_shapes = {
        impl_param.get_input_layout(0).get<ShapeType>(),
        impl_param.get_input_layout(1).get<ShapeType>(),
        impl_param.get_input_layout(2).get<ShapeType>(),
        impl_param.get_input_layout(3).get<ShapeType>(),
    };
    std::vector<ShapeType> output_shapes = ov::op::v1::shape_infer(&op, input_shapes);

    format output_format = format::adjust_to_rank(input0_layout.format, output_shapes[0].size());

    return { layout{output_shapes[0], output_type, output_format} };
}

template std::vector<layout> gather_tree_inst::calc_output_layouts<ov::PartialShape>(gather_tree_node const& node, const kernel_impl_params& impl_param);

std::string gather_tree_inst::to_string(gather_tree_node const& node) {
    std::stringstream primitive_description;
    node.desc_to_json()->dump(primitive_description);
    return primitive_description.str();
}

gather_tree_inst::typed_primitive_inst(network& network, gather_tree_node const& node) : parent(network, node) {
    auto dependencies = node.get_dependencies();

    for (auto& dep : dependencies) {
        if (dep.first->get_output_layout().is_dynamic()) {
            return;
        }
    }

    auto input_layout = node.get_input_layout();

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

    // check input dims
    CLDNN_ERROR_NOT_EQUAL(node.id(),
        "input0 size", dependencies.at(0).first->get_output_layout().get_tensor(), "output size", input_layout.get_tensor(),
        "mismatch");
    CLDNN_ERROR_NOT_EQUAL(node.id(),
        "input1 size", dependencies.at(1).first->get_output_layout().get_tensor(), "output size", input_layout.get_tensor(),
        "mismatch");
    CLDNN_ERROR_NOT_EQUAL(node.id(),
        "input2 size", dependencies.at(2).first->get_output_layout().count(), "node's feature size", input_layout.feature(),
        "There can't be more than one end_token");
    CLDNN_ERROR_NOT_EQUAL(node.id(),
        "input3 size", dependencies.at(3).first->get_output_layout().count(), "one", 1,
        "There can't be more than one end_token");
}
}  // namespace cldnn
