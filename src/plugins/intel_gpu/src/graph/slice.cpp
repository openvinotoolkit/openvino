// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <slice_inst.h>
#include "primitive_type_base.h"
#include "openvino/op/slice.hpp"
#include "slice_shape_inference.hpp"
#include <sstream>
#include <json_object.h>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(slice)

slice_inst::typed_primitive_inst(network& network, slice_node const& node)
    : parent(network, node) {}

layout slice_inst::calc_output_layout(slice_node const& node, kernel_impl_params const& impl_param) {
    return calc_output_layouts<ov::PartialShape>(node, impl_param)[0];
}

template<typename ShapeType>
inline std::vector<layout> slice_inst::calc_output_layouts(const slice_node&, const kernel_impl_params& impl_param) {
    std::vector<ShapeType> input_shapes{impl_param.input_layouts[0].get<ShapeType>()};
    std::unordered_map<size_t, ov::Tensor> const_data;
    const auto shape_len = input_shapes.back().rank().get_length();
    for (std::size_t i = 1; i < impl_param.input_layouts.size(); i++) {
        const ov::PartialShape input_shape{static_cast<ov::Dimension::value_type>(shape_len)};
        input_shapes.push_back(input_shape);
        if (impl_param.memory_deps.find(i) != impl_param.memory_deps.end()) {
            auto gpu_mem = impl_param.memory_deps.at(i);
            cldnn::mem_lock<uint8_t, mem_lock_type::read> gpu_mem_lock(gpu_mem, impl_param.get_stream());
            const_data.emplace(i, make_tensor(layout {input_shape, gpu_mem->get_layout().data_type, gpu_mem->get_layout().format },
                gpu_mem_lock.data()));
        }
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
    mem_lock<int32_t> lock(_shape_info_memory, _network.get_stream());
    auto shape_info_ptr = lock.data();
    size_t offset = 0;
    for (size_t i = 0; i < _node->get_dependencies().size(); i++) {
        if (i == 2)
            continue;
        GPU_DEBUG_TRACE_DETAIL << id() << " : update shape_info for input[" << i << "]" << std::endl;
        const auto& node_in_lay = _node->get_input_layout(i);
        const auto& runtime_in_lay = params.input_layouts[i];
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
