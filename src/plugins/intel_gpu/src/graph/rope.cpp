// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rope_inst.h"

#include "primitive_type_base.h"
#include "json_object.h"
#include <string>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(rope)

layout rope_inst::calc_output_layout(rope_node const& node, kernel_impl_params const& impl_param) {
    return calc_output_layouts<ov::PartialShape>(node, impl_param)[0];
}

template<typename ShapeType>
std::vector<layout> rope_inst::calc_output_layouts(rope_node const& node, kernel_impl_params const& impl_param) {
    auto desc = impl_param.typed_desc<rope>();

    auto input0_layout = impl_param.get_input_layout(0);
    auto input_pshape =  input0_layout.get<ShapeType>();
    auto output_format = input0_layout.format;

    auto output_type = desc->output_data_types[0].value_or(input0_layout.data_type);
    if (impl_param.has_fused_primitives()) {
        output_type = impl_param.get_fused_output_layout().data_type;
    }

    ShapeType output_shape = input_pshape;

    if (desc->config.is_qwen) {
        // Qwen specific RoPE
        // input  [batch_size, cur_length, (hidden_states_q + hidden_states_k + hidden_states_v)]
        // output [batch_size, cur_length, head_cnt, head_size]
        output_shape = {input_pshape[0], input_pshape[1], ov::Dimension(desc->config.head_cnt), ov::Dimension(desc->config.head_size)};
    } else if (desc->config.is_chatglm) {
        // chatGLM specific RoPE
        // input  [length, batch_size, (hidden_states_q + hidden_states_k + hidden_states_v)]
        // output [length, batch_size, head_cnt, hidden_states_k]
        output_shape = {input_pshape[0], input_pshape[1], ov::Dimension(desc->config.head_cnt), ov::Dimension(desc->config.head_size)};
        // mb last dim another <---------------------------------------------------------------------------------------------------------------
    } else {
        auto input_slice_size = desc->config.slice_stop - desc->config.slice_start;
        if (input_slice_size > 0) {
            output_shape[3] = input_slice_size;
        }
        if (desc->config.input_trans0213) {
            // transpose 0213 ([B,L,H,S]=>[B,H,L,S]) happens before RoPE
            std::swap(output_shape[2], output_shape[1]);
        } else if (desc->config.is_interleaved) {
            // transpose 0213 ([B,L,H,S]=>[B,H,L,S]) happens after RoPE
            std::swap(output_shape[2], output_shape[1]);
        }
    }
    return { layout{output_shape, output_type, output_format} };
}

template std::vector<layout> rope_inst::calc_output_layouts<ov::PartialShape>(rope_node const& node, const kernel_impl_params& impl_param);

std::string rope_inst::to_string(rope_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;

    json_composite rope_info;
    //rope_info.add("", );

    node_info->add("rope info", rope_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

}  // namespace cldnn
