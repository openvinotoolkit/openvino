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

    auto signal_layout = impl_param.get_input_layout(0);
    auto window_layout = impl_param.get_input_layout(1);
    auto frame_size_layout = impl_param.get_input_layout(2);
    auto frame_step_layout = impl_param.get_input_layout(3);

    std::vector<ShapeType> input_shapes = {
        signal_layout.get<ShapeType>(),
        window_layout.get<ShapeType>(),
        frame_size_layout.get<ShapeType>(),
        frame_step_layout.get<ShapeType>(),
    };

    std::vector<ShapeType> output_shapes;

    ov::op::v15::STFT op;
    op.set_transpose_frames(primitive->transpose_frames);
    output_shapes = shape_infer(&op, input_shapes);

    return {layout{output_shapes[0], signal_layout.data_type, signal_layout.format}};
}

std::string STFT_inst::to_string(STFT_node const& node) {
    auto node_info = node.desc_to_json();
    json_composite STFT_info;
    STFT_info.add("signal id", node.input(0).id());
    STFT_info.add("window id", node.input(1).id());
    STFT_info.add("transpose_frames", node.get_primitive()->transpose_frames);
    node_info->add("STFT info", STFT_info);
    std::stringstream primitive_description;
    node_info->dump(primitive_description);
    return primitive_description.str();
}

}  // namespace cldnn