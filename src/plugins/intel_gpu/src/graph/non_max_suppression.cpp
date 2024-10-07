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
std::vector<layout> non_max_suppression_gather_inst::calc_output_layouts(non_max_suppression_gather_node const& /*node*/,
                                                                         const kernel_impl_params& impl_param) {
    std::vector<layout> layouts;

    auto desc = impl_param.typed_desc<non_max_suppression_gather>();
    std::vector<ShapeType> output_shapes = { ShapeType{}, ShapeType{}, ShapeType{} };

    auto& memory_deps = impl_param.memory_deps;
    if (memory_deps.count(2)) {
        auto third_output = memory_deps.at(2);
        cldnn::mem_lock<int32_t, mem_lock_type::read> third_output_lock(third_output, impl_param.get_stream());
        auto third_output_data = third_output_lock.data();

        output_shapes[0] = ShapeType{third_output_data[0], 3};
    } else {
        output_shapes[0] = ShapeType{ov::Dimension::dynamic(), 3};
    }
    output_shapes[1] = output_shapes[0];
    output_shapes[2] = ShapeType{1};

    for (size_t i = 0; i < desc->num_outputs; ++i) {
        layouts.push_back({output_shapes[i],
                        impl_param.get_input_layout(i).data_type,
                        format::get_default_format(output_shapes[i].size())});
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

void non_max_suppression_gather_inst::on_execute() {
    update_output_memory();
}

void non_max_suppression_gather_inst::update_output_memory() {
    if (!can_be_optimized())
        return;

    for (size_t i = 0; i < inputs_memory_count(); i++) {
        if (node->get_program().is_new_shape_infer() && input_memory_ptr(i) == nullptr)
            return;

        if (output_memory_ptr(i) != nullptr && _network.get_engine().is_the_same_buffer(output_memory(i), input_memory(i)))
            return;

        // Can_be_optimized nodes are allocating from memory_pool too. In this case,
        // we need release the legacy output memory from memory pool explicitly.
        if (static_cast<bool>(_outputs[i]) &&
            _node->get_program().get_config().get_property(ov::intel_gpu::enable_memory_pool)) {
            _network.get_memory_pool().release_memory(_outputs[i].get(), _node->get_unique_id(), _node->id(), _network.get_id());
        }
        _outputs[i] = {_network.get_engine().reinterpret_buffer(input_memory(i), _impl_params->get_output_layout(i))};
    }
}

non_max_suppression_gather_inst::typed_primitive_inst(network& network, non_max_suppression_gather_node const& node) : parent(network, node) {}

}  // namespace cldnn
