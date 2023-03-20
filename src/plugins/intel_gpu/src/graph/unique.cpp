// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/unique.hpp"

#include <sstream>
#include <string>

#include "json_object.h"
#include "primitive_type_base.h"
#include "unique_inst.hpp"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(unique)

layout unique_inst::calc_output_layout(const unique_node& node, const kernel_impl_params& impl_param) {
    OPENVINO_THROW("Only calc_output_layouts should be used!");
}

template <typename ShapeType>
std::vector<layout> unique_inst::calc_output_layouts(const unique_node& node, const kernel_impl_params& impl_param) {
    std::vector<layout> layouts;
    const auto desc = impl_param.typed_desc<unique>();
    const auto input_layout = impl_param.get_input_layout();
    const auto input_shape = input_layout.get_partial_shape();

    // TODO: Properly calculate dynamic outputs
    std::vector<ShapeType> output_shapes = {ShapeType(), ShapeType(), ShapeType(), ShapeType()};
    output_shapes.at(0) = input_shape;

    if (desc->flattened) {
        const auto input_tensor_capacity = ov::shape_size(input_shape.to_shape());
        output_shapes.at(0) = ov::Shape{input_tensor_capacity};
    }

    for (size_t i = 0; i < desc->num_outputs; ++i) {
        const auto& output_shape = output_shapes.at(i);
        const auto output_dt = desc->output_data_types.at(i).value();
        layouts.push_back({output_shape, output_dt, format::get_default_format(output_shape.size())});
    }

    return layouts;
}

template std::vector<layout> unique_inst::calc_output_layouts<ov::PartialShape>(const unique_node& node,
                                                                                const kernel_impl_params& impl_param);

std::string unique_inst::to_string(const unique_node& node) {
    auto primitive = node.get_primitive();
    json_composite unique_info;
    unique_info.add("input", node.input().id());
    if (!primitive->flattened) {
        unique_info.add("axis", primitive->axis);
    }
    unique_info.add("sorted", primitive->sorted);

    auto node_info = node.desc_to_json();
    node_info->add("unique info", unique_info);

    std::ostringstream primitive_description;
    node_info->dump(primitive_description);
    return primitive_description.str();
}

}  // namespace cldnn
