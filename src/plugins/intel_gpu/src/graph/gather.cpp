// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gather_inst.h"

#include "primitive_type_base.h"
#include "json_object.h"
#include <string>

#include "gather_shape_inference.hpp"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(gather)

template<typename ShapeType>
std::vector<layout> gather_inst::calc_output_layouts(gather_node const& /*node*/, const kernel_impl_params& impl_param) {
    auto desc = impl_param.typed_desc<gather>();

    auto input0_layout = impl_param.get_input_layout(0);
    auto input1_layout = impl_param.get_input_layout(1);

    auto output_type = input0_layout.data_type;
    if (impl_param.typed_desc<gather>()->compressed_weights) {
        output_type = impl_param.typed_desc<gather>()->decompressed_type;
    }
    if (impl_param.has_fused_primitives()) {
        output_type = impl_param.get_output_element_type();
    }

    ov::op::v8::Gather op;
    op.set_batch_dims(desc->batch_dim);
    std::vector<ShapeType> output_shapes = {ShapeType()};
    std::vector<ShapeType> input_shapes = {
        input0_layout.get<ShapeType>(),
        input1_layout.get<ShapeType>(),
        ShapeType{1} // axis input is removed on gather primitive creation, so we can't use get_dependency(2)
    };

    int64_t axis = desc->axis;

    auto axis_tensor = ov::Tensor(ov::element::i64, ov::Shape{1}, static_cast<void*>(&axis));
    std::unordered_map<size_t, ov::Tensor> const_data = {{2, axis_tensor}};
    output_shapes = ov::op::shape_infer(&op, input_shapes, ov::make_tensor_accessor(const_data));

    format output_format = format::adjust_to_rank(input0_layout.format, output_shapes[0].size());

    return { layout{output_shapes[0], output_type, output_format} };
}

template std::vector<layout> gather_inst::calc_output_layouts<ov::PartialShape>(gather_node const& node, const kernel_impl_params& impl_param);

std::string gather_inst::to_string(gather_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto& input = node.input();

    std::stringstream primitive_description;

    json_composite gather_info;
    gather_info.add("input id", input.id());
    gather_info.add("axis", desc->axis);
    gather_info.add("batch_dim", desc->batch_dim);
    gather_info.add("output shape", cldnn::to_string(desc->output_shape));
    gather_info.add("compressed weights", desc->compressed_weights ? "true" : "false");
    if (desc->compressed_weights) {
        gather_info.add("decompression scale id", desc->decompression_scale.pid);
        gather_info.add("decompression zp id", desc->decompression_zero_point.pid);
        if (desc->decompression_zero_point_scalar.has_value()) {
            gather_info.add("decompression zp value", desc->decompression_zero_point_scalar.value());
        }
    }

    node_info->add("gather info", gather_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

void gather_inst::on_execute() {
    update_output_memory();
}

void gather_inst::update_output_memory() {
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

gather_inst::typed_primitive_inst(network& network, gather_node const& node) : parent(network, node) {}

}  // namespace cldnn
