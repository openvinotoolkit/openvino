// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <slice_inst.h>
#include "primitive_type_base.h"
#include "openvino/op/slice.hpp"
#include "slice_shape_inference.hpp"
#include <sstream>
#include <json_object.h>

namespace cldnn {

SliceKernelRefNeededInputs SliceKernelRefNeededInputs::Create(const slice_node& node) {
    SliceKernelRefNeededInputs inputs;

    const auto& node_inputs = node.get_dependencies();

    const bool axes_in_runtime =
        ((node_inputs.size() == InputIndices::kInputsNum) && !node_inputs[InputIndices::kAxes].first->is_constant());
    const bool start_in_runtime = !node_inputs[InputIndices::kStart].first->is_constant();
    const bool step_in_runtime = !node_inputs[InputIndices::kStep].first->is_constant();

    inputs.neededIndexes.push_back(InputIndices::kData);
    if (start_in_runtime)
        inputs.neededIndexes.push_back(InputIndices::kStart);
    if (step_in_runtime)
        inputs.neededIndexes.push_back(InputIndices::kStep);
    if (axes_in_runtime)
        inputs.neededIndexes.push_back(InputIndices::kAxes);

    // NOTE: stop is never needed as it is passed implicitely via output shape.

    return inputs;
}

GPU_DEFINE_PRIMITIVE_TYPE_ID(slice)

slice_inst::typed_primitive_inst(network& network, slice_node const& node)
    : parent(network, node) {}

layout slice_inst::calc_output_layout(slice_node const& node, kernel_impl_params const& impl_param) {
    return calc_output_layouts<ov::PartialShape>(node, impl_param)[0];
}

template <typename ShapeType>
inline std::vector<layout> slice_inst::calc_output_layouts(const slice_node&, const kernel_impl_params& impl_param) {
    std::vector<ShapeType> input_shapes{impl_param.input_layouts[0].get<ShapeType>()};
    std::unordered_map<size_t, ov::Tensor> const_data;
    for (std::size_t i = 1; i < impl_param.input_layouts.size(); i++) {
        // NOTE: This code effectively makes a reshape operation on tensors start,
        // stop, step and axes. The specification of Slice operator clearly says
        // that those tensors are 1D tensors - and this is what is expected
        // in shape_infer(). However, people in tests and other places,
        // put 4D tensors instead of 1D(e.g. [4,1,1,1] instead of [4]).
        // At the time of writing this comment - the hack for such situation
        // was already there, so adding an ASSERT will effectively make
        // some tests and graph transformations fail.
        // There should be some kind of warning to the user about it, but AFAIK
        // we don't have warning logs that could be enabled/disabled without
        // affecting performance...
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
    }
    ov::op::v8::Slice op;
    auto output_shapes = shape_infer(&op, input_shapes, ov::make_tensor_accessor(const_data));
    auto input_layout = impl_param.get_input_layout();
    std::vector<layout> output_layouts;
    for (size_t i = 0; i < output_shapes.size(); ++i) {
        output_layouts.push_back(layout({output_shapes[i], input_layout.data_type, input_layout.format}));
    }
    return output_layouts;
}

std::string slice_inst::to_string(slice_node const& node) {
    auto node_info = node.desc_to_json();
    json_composite slice_info;
    slice_info.add("input id", node.input().id());
    slice_info.add("begin_param id", node.get_dependency(1).id());
    slice_info.add("end_param id", node.get_dependency(2).id());
    slice_info.add("step_param id", node.get_dependency(3).id());
    slice_info.add("axis_param id", node.get_dependency(4).id());
    node_info->add("slice info", slice_info);
    std::stringstream primitive_description;
    node_info->dump(primitive_description);
    return primitive_description.str();
}

void slice_inst::update_shape_info_tensor(const kernel_impl_params& params) {
    if (!_shape_info_memory) {
        allocate_shape_info_memory();
    }

    mem_lock<int32_t> lock(_shape_info_memory, _network.get_stream());
    auto shape_info_ptr = lock.data();
    size_t offset = 0;
    const SliceKernelRefNeededInputs inputs = SliceKernelRefNeededInputs::Create(*_node);

    for (auto idx : inputs.GetNeededInputIndexes()) {
        GPU_DEBUG_TRACE_DETAIL << id() << " : update shape_info for input[" << idx << "]" << std::endl;
        const auto& node_in_lay = _node->get_input_layout(idx);
        const auto& runtime_in_lay = params.input_layouts[idx];
        fill_shape_info_data(runtime_in_lay, node_in_lay, shape_info_ptr, offset);
    }
    for (size_t i = 0; i < _node->get_output_layouts().size(); i++) {
        GPU_DEBUG_TRACE_DETAIL << id() << " : update shape_info for output[" << i << "]" << std::endl;
        const auto& node_out_lay = _node->get_output_layout(i);
        const auto& runtime_out_lay = params.output_layouts[i];
        fill_shape_info_data(runtime_out_lay, node_out_lay, shape_info_ptr, offset);
    }
}

} // namespace cldnn
