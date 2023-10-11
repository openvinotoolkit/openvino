// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "non_max_suppression_inst.h"
#include "primitive_type_base.h"
#include "json_object.h"
#include <string>

#include "nms_shape_inference.hpp"

namespace cldnn {
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

    ov::op::v9::NonMaxSuppression op;
    op.set_box_encoding(desc->center_point_box ? ov::op::v9::NonMaxSuppression::BoxEncodingType::CENTER
                                               : ov::op::v9::NonMaxSuppression::BoxEncodingType::CORNER);
    op.set_sort_result_descending(desc->sort_result_descending);

    std::vector<ShapeType> output_shapes = { ShapeType{}, ShapeType{}, ShapeType{} };
    std::vector<ShapeType> input_shapes = {
        impl_param.get_input_layout(0).get<ShapeType>(),
        impl_param.get_input_layout(1).get<ShapeType>()
    };

    auto& memory_deps = impl_param.memory_deps;
    std::unordered_map<size_t, ov::Tensor> const_data;
    if (memory_deps.count(2)) {
        auto max_output_boxes_per_class_mem = memory_deps.at(2);
        cldnn::mem_lock<uint8_t, mem_lock_type::read> max_output_boxes_per_class_lock(max_output_boxes_per_class_mem,
                                                                                      impl_param.get_stream());
        auto max_output_boxes_per_class_tensor = make_tensor(max_output_boxes_per_class_mem->get_layout(),
                                                                  max_output_boxes_per_class_lock.data());
        const_data.emplace(2, max_output_boxes_per_class_tensor);

        const auto& boxes = input_shapes[0];
        const auto& scores = input_shapes[1];
        // To produce a static output, we need to check dynamism of input tensor's dimensions
        // Output tensor has the following shape: [min(num_boxes, max_output_boxes_per_class) * num_batches * num_classes, 3]
        // The first dimension is an upper bound for the number of possible selected boxes
        bool static_output = boxes[1].is_static() && scores[0].is_static() && scores[1].is_static();
        output_shapes = ov::op::v9::shape_infer(&op, input_shapes, ov::make_tensor_accessor(const_data), static_output);
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

}  // namespace cldnn
