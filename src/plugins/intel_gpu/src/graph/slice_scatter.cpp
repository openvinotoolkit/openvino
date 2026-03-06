// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <slice_scatter_inst.h>
#include "primitive_type_base.h"
#include "openvino/op/slice_scatter.hpp"
#include "slice_scatter_shape_inference.hpp"
#include <sstream>
#include <json_object.h>

namespace cldnn {

SliceScatterKernelRefNeededInputs SliceScatterKernelRefNeededInputs::Create(const slice_scatter_node& node) {
    SliceScatterKernelRefNeededInputs inputs;

    const auto& node_inputs = node.get_dependencies();

    // data and updates are always needed
    inputs.neededIndexes.push_back(InputIndices::kData);
    inputs.neededIndexes.push_back(InputIndices::kUpdates);

    const bool start_in_runtime = !node_inputs[InputIndices::kStart].first->is_constant();
    const bool step_in_runtime = !node_inputs[InputIndices::kStep].first->is_constant();
    const bool axes_in_runtime =
        ((node_inputs.size() > InputIndices::kAxes) && !node_inputs[InputIndices::kAxes].first->is_constant());

    if (start_in_runtime)
        inputs.neededIndexes.push_back(InputIndices::kStart);
    if (step_in_runtime)
        inputs.neededIndexes.push_back(InputIndices::kStep);
    if (axes_in_runtime)
        inputs.neededIndexes.push_back(InputIndices::kAxes);

    // NOTE: stop/end is never needed at runtime as it is passed implicitly via the updates shape.

    return inputs;
}

bool SliceScatterKernelRefNeededInputs::IsInputNeededInRuntime(InputIndices type) const {
    for (const auto& idx : neededIndexes) {
        if (idx == static_cast<size_t>(type))
            return true;
    }
    return false;
}

GPU_DEFINE_PRIMITIVE_TYPE_ID(slice_scatter)

slice_scatter_inst::typed_primitive_inst(network& network, slice_scatter_node const& node) : parent(network, node) {
    update_output_memory();
}

void slice_scatter_inst::on_execute() {
    update_output_memory();

    // If output is not sharing buffer with data input (e.g. dynamic shapes or not optimized),
    // explicitly copy data to output before the kernel writes updates into slice positions.
    if (!_network.get_engine().is_the_same_buffer(output_memory(), input_memory())) {
        output_memory().copy_from(_network.get_stream(), input_memory());
    }
}

void slice_scatter_inst::update_output_memory() {
    if (!can_be_optimized() || _impl_params->is_dynamic())
        return;

    if (_outputs.size() > 0 && static_cast<bool>(_outputs[0])
        && _network.get_engine().is_the_same_buffer(output_memory(), input_memory()))
        return;

    build_deps();

    if (static_cast<bool>(_outputs[0]) &&
        get_node().get_program().get_config().get_enable_memory_pool()) {
        _network.get_memory_pool().release_memory(_outputs[0].get(), get_node().get_unique_id(), get_node().id(), _network.get_id());
    }
    _outputs = {_network.get_engine().reinterpret_buffer(input_memory(), _impl_params->get_output_layout())};
    _mem_allocated = false;
}

layout slice_scatter_inst::calc_output_layout(slice_scatter_node const& node, kernel_impl_params const& impl_param) {
    return calc_output_layouts<ov::PartialShape>(node, impl_param)[0];
}

template <typename ShapeType>
inline std::vector<layout> slice_scatter_inst::calc_output_layouts(const slice_scatter_node&, const kernel_impl_params& impl_param) {
    std::vector<ShapeType> input_shapes;
    std::unordered_map<size_t, ov::Tensor> const_data;

    for (std::size_t i = 0; i < impl_param.input_layouts.size(); i++) {
        if (i >= 2) {
            // Reshape 1D for start/stop/step/axes inputs (same as Slice)
            ov::PartialShape input_shape = ov::PartialShape::dynamic(1);
            if (impl_param.memory_deps.find(i) != impl_param.memory_deps.end()) {
                auto gpu_mem = impl_param.memory_deps.at(i);
                input_shape = {static_cast<ov::Dimension::value_type>(gpu_mem->count())};
                cldnn::mem_lock<uint8_t, mem_lock_type::read> gpu_mem_lock(gpu_mem, impl_param.get_stream());
                const_data.emplace(
                    i,
                    make_tensor(layout{input_shape, gpu_mem->get_layout().data_type, gpu_mem->get_layout().format},
                                gpu_mem_lock.data()));
            }
            input_shapes.push_back(input_shape);
        } else {
            input_shapes.push_back(impl_param.input_layouts[i].get<ShapeType>());
        }
    }

    ov::op::v15::SliceScatter op;
    auto output_shapes = shape_infer(&op, input_shapes, ov::make_tensor_accessor(const_data));
    auto input_layout = impl_param.get_input_layout();
    std::vector<layout> output_layouts;
    for (size_t i = 0; i < output_shapes.size(); ++i) {
        output_layouts.push_back(layout({output_shapes[i], input_layout.data_type, input_layout.format}));
    }
    return output_layouts;
}

std::string slice_scatter_inst::to_string(slice_scatter_node const& node) {
    auto node_info = node.desc_to_json();
    json_composite slice_scatter_info;
    slice_scatter_info.add("data", node.input(0).id());
    slice_scatter_info.add("updates", node.input(1).id());
    slice_scatter_info.add("start", node.get_dependency(2).id());
    slice_scatter_info.add("stop", node.get_dependency(3).id());
    slice_scatter_info.add("step", node.get_dependency(4).id());
    if (node.get_dependencies().size() > 5) {
        slice_scatter_info.add("axes", node.get_dependency(5).id());
    }
    node_info->add("slice_scatter info", slice_scatter_info);
    std::stringstream primitive_description;
    node_info->dump(primitive_description);
    return primitive_description.str();
}

void slice_scatter_inst::update_shape_info_tensor(const kernel_impl_params& params) {
    if (!_shape_info_memory) {
        allocate_shape_info_memory();
    }

    mem_lock<int32_t> lock(_shape_info_memory, _network.get_stream());
    auto shape_info_ptr = lock.data();
    size_t offset = 0;
    const SliceScatterKernelRefNeededInputs inputs = SliceScatterKernelRefNeededInputs::Create(*_node);

    for (auto idx : inputs.GetNeededInputIndexes()) {
        GPU_DEBUG_TRACE_DETAIL << id() << " : update shape_info for input[" << idx << "]" << std::endl;
        const auto& node_in_lay = get_node().get_input_layout(idx);
        const auto& runtime_in_lay = params.input_layouts[idx];
        fill_shape_info_data(runtime_in_lay, node_in_lay, shape_info_ptr, offset);
    }
    for (size_t i = 0; i < get_node().get_output_layouts().size(); i++) {
        GPU_DEBUG_TRACE_DETAIL << id() << " : update shape_info for output[" << i << "]" << std::endl;
        const auto& node_out_lay = get_node().get_output_layout(i);
        const auto& runtime_out_lay = params.output_layouts[i];
        fill_shape_info_data(runtime_out_lay, node_out_lay, shape_info_ptr, offset);
    }
}

}  // namespace cldnn
