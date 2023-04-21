// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/unique.hpp"

#include <sstream>
#include <string>

#include "intel_gpu/runtime/memory.hpp"
#include "json_object.h"
#include "primitive_type_base.h"
#include "unique_inst.hpp"

namespace cldnn {

// -----------------------------------------------
// unique
// -----------------------------------------------
GPU_DEFINE_PRIMITIVE_TYPE_ID(unique)

layout unique_inst::calc_output_layout(const unique_node& node, const kernel_impl_params& impl_param) {
    OPENVINO_THROW("Only calc_output_layouts should be used!");
}

template <typename ShapeType>
std::vector<layout> unique_inst::calc_output_layouts(const unique_node& node, const kernel_impl_params& impl_param) {
    return {layout{ov::PartialShape{1}, cldnn::data_types::i64, cldnn::format::bfyx}};
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

    auto node_info = node.desc_to_json();
    node_info->add("unique info", unique_info);

    std::ostringstream primitive_description;
    node_info->dump(primitive_description);
    return primitive_description.str();
}

// -----------------------------------------------
// unique_reshape
// -----------------------------------------------
GPU_DEFINE_PRIMITIVE_TYPE_ID(unique_reshape)

layout unique_reshape_inst::calc_output_layout(const unique_reshape_node& node, const kernel_impl_params& impl_param) {
    OPENVINO_THROW("Only calc_output_layouts should be used!");
}

template <typename ShapeType>
std::vector<layout> unique_reshape_inst::calc_output_layouts(const unique_reshape_node& node,
                                                             const kernel_impl_params& impl_param) {
    std::vector<layout> layouts;
    const auto desc = impl_param.typed_desc<unique_reshape>();
    const auto input_layout = impl_param.get_input_layout();

    std::vector<ShapeType> output_shapes = {ShapeType(), ShapeType(), ShapeType(), ShapeType()};

    if (!impl_param.memory_deps.count(1)) {
        if (desc->flattened) {
            output_shapes.at(0) = ov::PartialShape{ov::Dimension::dynamic()};
        } else {
            output_shapes.at(0) = ov::PartialShape::dynamic(input_layout.get_partial_shape().rank());
        }
        output_shapes.at(1) = ov::PartialShape{ov::Dimension::dynamic()};
        output_shapes.at(2) = ov::PartialShape{ov::Dimension::dynamic()};
        output_shapes.at(3) = ov::PartialShape{ov::Dimension::dynamic()};
    } else {
        const auto input_shape = input_layout.get_shape();
        const size_t unique_count = read_vector<int64_t>(impl_param.memory_deps.at(1), impl_param.get_stream()).at(0);
        if (desc->flattened) {
            const auto input_tensor_capacity = ov::shape_size(input_shape);
            output_shapes.at(0) = ov::Shape{unique_count};
            output_shapes.at(1) = ov::Shape{unique_count};
            output_shapes.at(2) = ov::Shape{input_tensor_capacity};
            output_shapes.at(3) = ov::Shape{unique_count};
        } else {
            auto output_shape = input_shape;
            auto& new_axis_dimension = output_shape.at(desc->axis);
            const auto old_axis_dimension = new_axis_dimension;
            new_axis_dimension = unique_count;
            output_shapes.at(0) = output_shape;
            output_shapes.at(1) = ov::Shape{new_axis_dimension};
            output_shapes.at(2) = ov::Shape{old_axis_dimension};
            output_shapes.at(3) = ov::Shape{new_axis_dimension};
        }
    }

    for (auto i = 0U; i < desc->num_outputs; ++i) {
        const auto& output_shape = output_shapes.at(i);
        const auto output_dt = desc->output_data_types.at(i).value();
        auto output_format = format::get_default_format(output_shape.size());
        if (i == 0) {
            if (desc->flattened) {
                output_format = format::adjust_to_rank(input_layout.format, output_shape.size());
            } else {
                output_format = input_layout.format;
            }
        }
        layouts.emplace_back(output_shape, output_dt, output_format);
    }

    return layouts;
}

template std::vector<layout> unique_reshape_inst::calc_output_layouts<ov::PartialShape>(
    const unique_reshape_node& node,
    const kernel_impl_params& impl_param);

std::string unique_reshape_inst::to_string(const unique_reshape_node& node) {
    auto primitive = node.get_primitive();
    json_composite unique_reshape_info;
    unique_reshape_info.add("input", node.input().id());
    if (!primitive->flattened) {
        unique_reshape_info.add("axis", primitive->axis);
    }
    unique_reshape_info.add("sorted", primitive->sorted);

    auto node_info = node.desc_to_json();
    node_info->add("unique_reshape info", unique_reshape_info);

    std::ostringstream primitive_description;
    node_info->dump(primitive_description);
    return primitive_description.str();
}

}  // namespace cldnn
