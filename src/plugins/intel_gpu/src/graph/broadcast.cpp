// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "broadcast_inst.h"
#include "broadcast_shape_inference.hpp"

#include "json_object.h"
#include "primitive_type_base.h"
#include <string>
#include <vector>
#include <set>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(broadcast)

template<typename ShapeType>
std::vector<layout> broadcast_inst::calc_output_layouts(broadcast_node const& /*node*/, const kernel_impl_params& impl_param) {
    auto desc = impl_param.typed_desc<broadcast>();
    auto input0_layout = impl_param.get_input_layout(0);

    auto output_type = input0_layout.data_type;
    if (impl_param.has_fused_primitives()) {
        output_type = impl_param.get_output_element_type();
    }


    ov::op::v3::Broadcast op;
    op.set_broadcast_spec(desc->broadcast_mode);
    bool third_input_needed = desc->broadcast_mode == ov::op::BroadcastType::EXPLICIT;
    auto target_shape = desc->target_shape;

    ShapeType pattern_shape = impl_param.input_layouts.size() == 2 ? impl_param.get_input_layout(1).get<ShapeType>()
                                                                   : ShapeType(ov::Shape{ target_shape.size() });
    std::vector<ShapeType> output_shapes = {ShapeType{}};
    std::vector<ShapeType> input_shapes = {
        input0_layout.get<ShapeType>(),
        pattern_shape
    };

    auto axes_mapping = desc->axes_mapping.to_vector();
    ShapeType axes_mapping_shape = ov::Shape{axes_mapping.size()};

    std::unordered_map<size_t, ov::Tensor> const_data;
    if (third_input_needed) {
        input_shapes.emplace_back(axes_mapping_shape);

        auto axes_mapping_tensor = make_tensor({axes_mapping_shape, data_types::i64, format::bfyx},
                                                    static_cast<void*>(axes_mapping.data()));
        const_data.emplace(2, axes_mapping_tensor);
    }

    auto& constant_mem = impl_param.memory_deps;
    if (constant_mem.count(1)) {
        auto target_shape_mem = constant_mem.at(1);
        cldnn::mem_lock<uint8_t, mem_lock_type::read> target_shape_lock(target_shape_mem, impl_param.get_stream());
        const_data.emplace(1, make_tensor(target_shape_mem->get_layout(), target_shape_lock.data()));
        output_shapes = ov::op::v3::shape_infer(&op, input_shapes, ov::make_tensor_accessor(const_data));
    } else if (impl_param.input_layouts.size() == 1) {
        // predefined pattern shape
        if (target_shape.empty()) {
            target_shape.push_back(0); // add some value to vec to have not null ptr in tensor
        }
        auto target_shape_tensor = make_tensor({pattern_shape, data_types::i64, format::bfyx}, static_cast<void*>(target_shape.data()));
        const_data.emplace(1, target_shape_tensor);
        output_shapes = ov::op::v3::shape_infer(&op, input_shapes, ov::make_tensor_accessor(const_data));
    } else if (impl_param.input_layouts.size() >= 2) {
        auto input1 = impl_param.get_input_layout(1);
        auto output_rank = input1.get<ShapeType>().size();
        if (input1.is_static()) {
            output_rank = input1.get_dim(0);    // target shape rank is set as second input.
        }
        output_shapes[0] = desc->output_pshape.rank().is_static() ? desc->output_pshape : ShapeType::dynamic(std::max(static_cast<int>(output_rank), 1));
    }

    format output_format = format::adjust_to_rank(input0_layout.format, output_shapes[0].size());

    return { layout{output_shapes[0], output_type, output_format} };
}

template std::vector<layout> broadcast_inst::calc_output_layouts<ov::PartialShape>(broadcast_node const& node, const kernel_impl_params& impl_param);

std::string broadcast_inst::to_string(broadcast_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto& input = node.input();

    std::stringstream primitive_description;
    std::stringstream ss_broadcast_axes;
    ss_broadcast_axes << desc->axes_mapping;
    json_composite broadcast_info;
    broadcast_info.add("input id", input.id());
    broadcast_info.add("broadcast axes", ss_broadcast_axes.str());

    node_info->add("broadcast info", broadcast_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

void broadcast_inst::on_execute() {
    update_output_memory();
}

void broadcast_inst::update_output_memory() {
    if (!can_be_optimized())
        return;
    if (static_cast<bool>(_outputs[0]) && _network.get_engine().is_the_same_buffer(output_memory(), input_memory()))
        return;

    if (_node != nullptr)
        build_deps();

    GPU_DEBUG_TRACE_DETAIL << id() << " : update_output_memory with mem of input " << get_node().get_dependency(0).id()
                           << " : " << input_memory_ptr()->buffer_ptr() << std::endl;
    // Can_be_optimized nodes are allocating from memory_pool too. In this case,
    // we need release the legacy output memory from memory pool explicitly.
    if (static_cast<bool>(_outputs[0]) &&
        _node->get_program().get_config().get_property(ov::intel_gpu::enable_memory_pool)) {
        _network.get_memory_pool().release_memory(_outputs[0].get(), _node->get_unique_id(), _node->id(), _network.get_id());
    }
    _outputs[0] = input_memory_ptr();
    _mem_allocated = false;
}

broadcast_inst::typed_primitive_inst(network& network, broadcast_node const& node) : parent(network, node) { }
}  // namespace cldnn
