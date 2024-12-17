// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <json_object.h>
#include <stft_inst.h>

#include <sstream>

#include "openvino/core/enum_names.hpp"
#include "primitive_type_base.h"
#include "stft_shape_inference.hpp"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(STFT)

STFT_inst::typed_primitive_inst(network& network, STFT_node const& node) : parent(network, node) {}

layout STFT_inst::calc_output_layout(STFT_node const& node, kernel_impl_params const& impl_param) {
    return calc_output_layouts<ov::PartialShape>(node, impl_param)[0];
}

template <typename ShapeType>
std::vector<layout> STFT_inst::calc_output_layouts(STFT_node const& node, kernel_impl_params const& impl_param) {
    auto primitive = impl_param.typed_desc<STFT>();

    const auto& signal_layout = impl_param.get_input_layout(0);
    const auto& window_layout = impl_param.get_input_layout(1);
    const auto& frame_size_layout = impl_param.get_input_layout(2);
    const auto& frame_step_layout = impl_param.get_input_layout(3);

    std::vector<ShapeType> input_shapes = {
        signal_layout.get<ShapeType>(),
        window_layout.get<ShapeType>(),
        frame_size_layout.get<ShapeType>(),
        frame_step_layout.get<ShapeType>(),
    };

    std::unordered_map<size_t, ov::Tensor> const_data;
    for (std::size_t i = 2; i < impl_param.input_layouts.size(); ++i) {
        ov::PartialShape input_shape = ov::PartialShape(input_shapes[i]);
        if (impl_param.memory_deps.find(i) != impl_param.memory_deps.end()) {
            auto gpu_mem = impl_param.memory_deps.at(i);
            const auto count = static_cast<ov::Dimension::value_type>(gpu_mem->count());
            OPENVINO_ASSERT(count == 1, "The shape of frame_size and frame_step must be a scalar.");
            cldnn::mem_lock<uint8_t, mem_lock_type::read> gpu_mem_lock(gpu_mem, impl_param.get_stream());
            const_data.emplace(
                i,
                make_tensor(layout{input_shape, gpu_mem->get_layout().data_type, gpu_mem->get_layout().format},
                            gpu_mem_lock.data()));
        }
    }

    std::vector<ShapeType> output_shapes;
    ov::op::v15::STFT op;
    op.set_transpose_frames(primitive->transpose_frames);
    output_shapes = shape_infer(&op, input_shapes, ov::make_tensor_accessor(const_data));

    return {layout{output_shapes[0], signal_layout.data_type, signal_layout.format}};
}

std::string STFT_inst::to_string(STFT_node const& node) {
    auto node_info = node.desc_to_json();
    json_composite STFT_info;
    STFT_info.add("signal", node.input(0).id());
    STFT_info.add("window", node.input(1).id());
    STFT_info.add("framesize", node.input(2).id());
    STFT_info.add("framestep", node.input(3).id());
    STFT_info.add("transpose_frames", node.get_primitive()->transpose_frames);
    node_info->add("STFT info", STFT_info);
    std::stringstream primitive_description;
    node_info->dump(primitive_description);
    return primitive_description.str();
}

}  // namespace cldnn