// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "non_max_suppression_inst.h"
#include "primitive_type_base.h"
#include "json_object.h"
#include <string>

#include "intel_gpu/runtime/tensor_accessor.hpp"
#include "openvino/op/nms_rotated.hpp"
#include "nms_shape_inference.hpp"

namespace cldnn {

// -----------------------------------------------
// non_max_suppression
// -----------------------------------------------
GPU_DEFINE_PRIMITIVE_TYPE_ID(non_max_suppression)

layout non_max_suppression_inst::calc_output_layout(non_max_suppression_node const& node, kernel_impl_params const& impl_param) {
    auto desc = impl_param.typed_desc<non_max_suppression>();

    auto output_type = desc->output_data_types[0].value_or(data_types::i32);

    auto output_size = tensor(batch(desc->selected_indices_num), feature(3));
    return layout(output_type, impl_param.get_input_layout().format, output_size);
}

template<typename ShapeType>
std::vector<layout> non_max_suppression_inst::calc_output_layouts(non_max_suppression_node const& /*node*/, const kernel_impl_params& impl_param) {
    std::vector<layout> layouts;

    auto desc = impl_param.typed_desc<non_max_suppression>();

    TensorsContainer const_data(&impl_param.get_stream(), impl_param.memory_deps);
    std::vector<ShapeType> output_shapes = { ShapeType{}, ShapeType{}, ShapeType{} };
    std::vector<ShapeType> input_shapes = {
        impl_param.get_input_layout(0).get<ShapeType>(),
        impl_param.get_input_layout(1).get<ShapeType>()
    };

    const auto& boxes = input_shapes[0];
    const auto& scores = input_shapes[1];
    // To produce a static output, we need to check dynamism of input tensor's dimensions
    // Output tensor has the following shape: [min(num_boxes, max_output_boxes_per_class) * num_batches * num_classes, 3]
    // The first dimension is an upper bound for the number of possible selected boxes
    bool static_output = boxes[1].is_static() && scores[0].is_static() && scores[1].is_static();

    if (desc->rotation != non_max_suppression::Rotation::NONE) {
        ov::op::v13::NMSRotated op;
        op.set_clockwise(desc->rotation == non_max_suppression::Rotation::CLOCKWISE);
        op.set_sort_result_descending(desc->sort_result_descending);

        output_shapes = ov::op::v13::shape_infer(&op, input_shapes, cldnn::make_tensor_accessor(const_data), static_output);
    } else {
        ov::op::v9::NonMaxSuppression op;
        op.set_box_encoding(desc->center_point_box ? ov::op::v9::NonMaxSuppression::BoxEncodingType::CENTER
                                                   : ov::op::v9::NonMaxSuppression::BoxEncodingType::CORNER);
        op.set_sort_result_descending(desc->sort_result_descending);

        output_shapes = ov::op::v9::shape_infer(&op, input_shapes, cldnn::make_tensor_accessor(const_data), static_output);
    }

    for (size_t i = 0; i < desc->num_outputs; ++i) {
        auto dt = desc->output_data_types[i].value_or(data_types::i32);
        layouts.push_back({output_shapes[i], dt, format::get_default_format(output_shapes[i].size())});
    }
    return layouts;
}

template std::vector<layout> non_max_suppression_inst::calc_output_layouts<ov::PartialShape>(non_max_suppression_node const& node,
                                                                                             const kernel_impl_params& impl_param);

std::string non_max_suppression_inst::to_string(non_max_suppression_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();

    json_composite info;
    info.add("center point box", desc->center_point_box);

    node_info->add("non max suppression info", info);

    std::stringstream description;
    node_info->dump(description);
    return description.str();
}

// -----------------------------------------------
// non_max_suppression_gather
// -----------------------------------------------
GPU_DEFINE_PRIMITIVE_TYPE_ID(non_max_suppression_gather)

layout non_max_suppression_gather_inst::calc_output_layout(non_max_suppression_gather_node const& node, kernel_impl_params const& impl_param) {
    OPENVINO_THROW("Only calc_output_layouts should be used!");
}

template<typename ShapeType>
std::vector<layout> non_max_suppression_gather_inst::calc_output_layouts(non_max_suppression_gather_node const& node, const kernel_impl_params& impl_param) {
    std::vector<layout> layouts;

    auto desc = impl_param.typed_desc<non_max_suppression_gather>();
    const auto input0_layout = node.get_dependency(0).get_output_layout(0);     // impl_param.get_input_layout(0);
    const auto input1_layout = node.get_dependency(0).get_output_layout(1);     // impl_param.get_input_layout(1);

    std::vector<ShapeType> output_shapes = { ShapeType{}, ShapeType{}, ShapeType{} };

    auto& memory_deps = impl_param.memory_deps;
    if (memory_deps.count(0)) {
        auto actual_output = memory_deps.at(0);
        cldnn::mem_lock<int32_t, mem_lock_type::read> actual_output_lock(actual_output, impl_param.get_stream());

        auto output_ps = actual_output->get_layout().get_partial_shape();
        auto b = output_ps[0].get_length();
        auto f = output_ps[1].get_length();  // should be 3

        // find valid data size
        auto output_data = actual_output_lock.data();
        int64_t actual_valid_num = b;
        for (int64_t i = 0; i < b ; i += 1) {
            if (output_data[i * f] == -1) {
                actual_valid_num = i;
                break;
            }
        }

        output_shapes[0] = output_shapes[1] = ShapeType{actual_valid_num, f};
        output_shapes[2] = ShapeType{1};
    } else {
        output_shapes[0] = output_shapes[1] = ShapeType{ov::Dimension::dynamic(), 3};
        output_shapes[2] = ShapeType{1};
    }

    for (size_t i = 0; i < desc->num_outputs; ++i) {
        auto dt = desc->output_data_types[i].value_or(data_types::i32);
        layouts.push_back({output_shapes[i], dt, format::get_default_format(output_shapes[i].size())});
    }
    return layouts;
}

template std::vector<layout> non_max_suppression_gather_inst::calc_output_layouts<ov::PartialShape>(non_max_suppression_gather_node const& node,
                                                                                             const kernel_impl_params& impl_param);

std::string non_max_suppression_gather_inst::to_string(non_max_suppression_gather_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();

    json_composite info;

    node_info->add("non max suppression gather info", info);

    std::stringstream description;
    node_info->dump(description);
    return description.str();
}

}  // namespace cldnn
