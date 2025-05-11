// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_ops/glu.hpp"
#include "glu_shape_inference.hpp"
#include "swiglu_inst.h"

#include "primitive_type_base.h"
#include "json_object.h"
#include <string>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(swiglu);

layout swiglu_inst::calc_output_layout(swiglu_node const& node, kernel_impl_params const& impl_param) {
    auto desc = impl_param.typed_desc<swiglu>();
    auto input_layout = impl_param.get_input_layout();
    auto output_type = impl_param.desc->output_data_types[0].value_or(input_layout.data_type);
    auto output_format = input_layout.format;

    return layout(output_type, output_format, desc->output_size);
}

template<typename ShapeType>
std::vector<layout> swiglu_inst::calc_output_layouts(swiglu_node const& /*node*/, const kernel_impl_params& impl_param) {
    auto desc = impl_param.typed_desc<swiglu>();
    auto input_layout = impl_param.get_input_layout();
    auto output_type = impl_param.desc->output_data_types[0].value_or(input_layout.data_type);
    auto output_format = input_layout.format;

    ov::op::internal::GLU op;
    op.set_axis(desc->axis);
    op.set_split_lengths(desc->split_lengths);

    std::vector<ShapeType> input_shapes = {impl_param.get_input_layout(0).get<ShapeType>()};

    std::vector<ShapeType> output_shapes = shape_infer(&op, input_shapes);

    return { layout(output_shapes[0], output_type, output_format) };
}

template std::vector<layout> swiglu_inst::calc_output_layouts<ov::PartialShape>(swiglu_node const& node,
                                                                                const kernel_impl_params& impl_param);

std::string swiglu_inst::to_string(swiglu_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto& input = node.input();

    std::stringstream primitive_description;

    json_composite swiglu_info;
    swiglu_info.add("input_id", input.id());
    swiglu_info.add("axis", desc->axis);
    swiglu_info.add("split_lengths", desc->split_lengths);

    node_info->add("swiglu_info", swiglu_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

swiglu_inst::typed_primitive_inst(network& network, swiglu_node const& node) : parent(network, node) {}

}  // namespace cldnn
