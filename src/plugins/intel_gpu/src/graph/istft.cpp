// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <istft_inst.h>
#include <json_object.h>

#include <sstream>

#include "istft_shape_inference.hpp"
#include "memory_accessor.hpp"
#include "openvino/core/enum_names.hpp"
#include "primitive_type_base.h"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(ISTFT)

ISTFT_inst::typed_primitive_inst(network& network, ISTFT_node const& node) : parent(network, node) {}

layout ISTFT_inst::calc_output_layout(ISTFT_node const& node, kernel_impl_params const& impl_param) {
    return calc_output_layouts<ov::PartialShape>(node, impl_param)[0];
}

template <typename ShapeType>
std::vector<layout> ISTFT_inst::calc_output_layouts(ISTFT_node const& node, kernel_impl_params const& impl_param) {
    auto primitive = impl_param.typed_desc<ISTFT>();

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

    if (impl_param.input_layouts.size() == 5) {
        const auto& length_layout = impl_param.get_input_layout(4);
        input_shapes.push_back(length_layout.get<ShapeType>());
    }

    const auto ta = MemoryAccessor(&impl_param.memory_deps, impl_param.get_stream());

    std::vector<ShapeType> output_shapes;
    ov::op::v16::ISTFT op;
    op.set_center(primitive->center);
    op.set_normalized(primitive->normalized);
    output_shapes = shape_infer(&op, input_shapes, ta);

    return {layout{output_shapes[0], signal_layout.data_type, signal_layout.format}};
}

std::string ISTFT_inst::to_string(ISTFT_node const& node) {
    auto node_info = node.desc_to_json();
    json_composite ISTFT_info;
    ISTFT_info.add("signal", node.input(0).id());
    ISTFT_info.add("window", node.input(1).id());
    ISTFT_info.add("framesize", node.input(2).id());
    ISTFT_info.add("framestep", node.input(3).id());
    ISTFT_info.add("center", node.get_primitive()->center);
    ISTFT_info.add("normalized", node.get_primitive()->normalized);
    node_info->add("ISTFT info", ISTFT_info);
    std::stringstream primitive_description;
    node_info->dump(primitive_description);
    return primitive_description.str();
}

}  // namespace cldnn