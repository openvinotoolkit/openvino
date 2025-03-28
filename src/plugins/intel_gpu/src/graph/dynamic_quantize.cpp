// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_ops/dynamic_quantize.hpp"
#include "dynamic_quantize_inst.h"
#include "fully_connected_inst.h"

#include "primitive_type_base.h"
#include "json_object.h"
#include <string>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(dynamic_quantize);

static bool should_skip_execution(dynamic_quantize_node const& node, const layout &act_layout) {
    // std::cout << __LINE__ << "  " << node.is_runtime_skippable() << "  " << act_layout.is_static() << "  " << act_layout.get_partial_shape() << std::endl;
    if (!node.is_runtime_skippable()
        || !act_layout.is_static())
        return false;

    // XXX: need to use global configuration
    // GPU_DEBUG_IF (get_config().get_disable_dynamic_quantization_opt()) {
    //     return false;
    // }

    // Do not skip dynamic quantization if next node is not fully connected.(such as SDPA)
    if (!(*node.get_users().begin())->is_type<fully_connected>())
        return false;

    // If batch size is small, dynamic_quantize is disabled for performance reason
    size_t input_batch = act_layout.batch();
    // 3D input
    if (act_layout.format == format::bfyx) {
        input_batch = act_layout.batch() * act_layout.feature();
    }

    if (input_batch <= 1) {
        GPU_DEBUG_TRACE_DETAIL << "[" << __func__ << "]"
                               << "  can_be_optimized - " << node.id() << " - " << act_layout.get_shape() << std::endl;
        // XXX: need to restore this assertion
        // OPENVINO_ASSERT(node.get_user_insts().size() == node.get_outputs_count(), "Dynamic quantization is supposed to have only one user-node with duplicated connection: ", get_node().id());
        return true;
    }
    return false;
}

layout dynamic_quantize_inst::calc_output_layout(dynamic_quantize_node const& /*node*/, kernel_impl_params const& impl_param) {
    auto desc = impl_param.typed_desc<dynamic_quantize>();
    const auto& input_layout = impl_param.get_input_layout();
    auto output_type = data_types::i8;
    auto output_format = input_layout.format;

    return layout(output_type, output_format, input_layout.get_tensor());
}

template<typename ShapeType>
std::vector<layout> dynamic_quantize_inst::__calc_output_layouts(dynamic_quantize_node const& node,
                                                                 const layout &act_layout,
                                                                 const dynamic_quantize::Attributes& attrs) {
    ov::op::internal::DynamicQuantize op;
    op.set_attrs(attrs);

    auto output_format = act_layout.format;

    std::vector<ShapeType> input_shapes = {
        act_layout.get<ShapeType>(),
    };
    // std::cout << act_layout << std::endl;

    auto output_shapes = ov::op::internal::DynamicQuantize::shape_infer(&op, input_shapes);

    std::vector<layout> output_layouts = {  layout(output_shapes[0], attrs.quantization_dt, output_format),
                                            layout(output_shapes[1], attrs.scale_dt, output_format) };

    if (should_skip_execution(node, act_layout)) {
        // std::cout << "should skip execution " << node.id() << std::endl;
        output_layouts[0] = act_layout;
    } else { 
        // std::cout << "do not skip execution " << node.id() << " - " << output_layouts[0].data_type << std::endl;
    }
        
    if (attrs.quantization_type == ov::op::internal::DynamicQuantize::QuantizationType::Asymmetric &&
        attrs.output_storage_type == ov::op::internal::DynamicQuantize::OutputStorageType::Planar) {
        output_layouts.emplace_back(layout(output_shapes[2], attrs.zp_dt, output_format));
    }

    return output_layouts;
}

template std::vector<layout> dynamic_quantize_inst::__calc_output_layouts<ov::PartialShape>(dynamic_quantize_node const& node,
                                                                                            const layout &act_layout,
                                                                                            const dynamic_quantize::Attributes& config);

template<typename ShapeType>
std::vector<layout> dynamic_quantize_inst::calc_output_layouts(dynamic_quantize_node const& node, const kernel_impl_params& impl_param) {
    auto desc = impl_param.typed_desc<dynamic_quantize>();
    const auto& input_layout = impl_param.get_input_layout();

    return __calc_output_layouts<ov::PartialShape>(node, input_layout, desc->attrs);
}

template std::vector<layout> dynamic_quantize_inst::calc_output_layouts<ov::PartialShape>(dynamic_quantize_node const& node,
                                                                                const kernel_impl_params& impl_param);

std::string dynamic_quantize_inst::to_string(dynamic_quantize_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;

    json_composite dynamic_quantize_info;
    dynamic_quantize_info.add("output_storage_type", static_cast<int>(desc->attrs.output_storage_type));
    dynamic_quantize_info.add("scales_zp_output_order", desc->attrs.scales_zp_output_order);
    dynamic_quantize_info.add("group_sizes", desc->attrs.group_sizes);
    dynamic_quantize_info.add("quantization_dt", desc->attrs.quantization_dt);
    dynamic_quantize_info.add("scale_dt", desc->attrs.scale_dt);
    dynamic_quantize_info.add("zp_dt", desc->attrs.zp_dt);
    dynamic_quantize_info.add("quantization_type", static_cast<int>(desc->attrs.quantization_type));
    node_info->add("dynamic_quantize info", dynamic_quantize_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

dynamic_quantize_inst::typed_primitive_inst(network& network, dynamic_quantize_node const& node) : parent(network, node) {}

void dynamic_quantize_inst::on_execute() {
    update_output_memory();
}

void dynamic_quantize_inst::update_output_memory() {
    if (!can_be_optimized())
        return;

    if (static_cast<bool>(_outputs[0])
        && _network.get_engine().is_the_same_buffer(output_memory(), input_memory())
        && output_memory().get_layout().identical(get_output_layout()))
        return;

    if (_node != nullptr)
        build_deps();

    // Do not update output memory when dynamic_quantize is optimized out
    // but input memory is not allocated yet because input is dynamic.
    // Since dep's _outputs may be empty, Check whether input memory is null by dep's outputs_allocated()
    OPENVINO_ASSERT(dependencies().front().first->outputs_allocated(), "[GPU] Dynamic quantize is optimized out, but its predecessor does not have output buffer.");

    // Can_be_optimized nodes are allocating from memory_pool too. In this case,
    // we need release the legacy output memory from memory pool explicitly.
    if (static_cast<bool>(_outputs[0]) &&
        _node->get_program().get_config().get_enable_memory_pool()) {
        _network.get_memory_pool().release_memory(_outputs[0].get(), _node->get_unique_id(), _node->id(), _network.get_id());
    }

    _outputs[0] = input_memory_ptr();
    _mem_allocated = false;
}
}  // namespace cldnn
