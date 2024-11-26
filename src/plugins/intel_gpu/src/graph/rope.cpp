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

    const auto& input0_layout = impl_param.get_input_layout(0);
    const auto& input0_shape = input0_layout.get<ShapeType>();
    auto output_format = input0_layout.format;

    auto output_type = desc->output_data_types[0].value_or(input0_layout.data_type);
    if (impl_param.has_fused_primitives()) {
        output_type = impl_param.get_output_element_type();
    }

    ShapeType output_shape = input0_shape;

    if (desc->config.is_qwen) {
        output_shape = { input0_shape[0],
                         input0_shape[1],
                         ov::Dimension(desc->config.head_cnt),
                         ov::Dimension(desc->config.head_size) };
    } else if (desc->config.is_chatglm) {
        if (desc->config.support_2d_rope) {
            // input0_shape = [batch_size, seq_length]
            output_shape = { input0_shape[0],
                            ov::Dimension(desc->config.head_cnt),
                            input0_shape[1],
                            ov::Dimension(desc->config.head_size) };
        } else {
            output_shape = { input0_shape[0],
                            input0_shape[1],
                            ov::Dimension(desc->config.head_cnt),
                            ov::Dimension(desc->config.head_size) };
        }
    } else {
        auto input_slice_size = desc->config.slice_stop - desc->config.slice_start;
        if (input_slice_size > 0) {
            output_shape[3] = input_slice_size;
        }

        if (desc->config.input_trans0213 || desc->config.output_trans0213) {
            std::swap(output_shape[2], output_shape[1]);
        }
    }
    return { layout(output_shape, output_type, output_format) };
}

template std::vector<layout> rope_inst::calc_output_layouts<ov::PartialShape>(rope_node const& node, const kernel_impl_params& impl_param);

std::string rope_inst::to_string(rope_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;

    json_composite rope_info;
    rope_info.add("gather_position_arg_id", desc->config.gather_position_arg_id);
    rope_info.add("gather_rank", desc->gather_rank);
    rope_info.add("head_cnt", desc->config.head_cnt);
    rope_info.add("head_size", desc->config.head_size);
    rope_info.add("input_trans0213", desc->config.input_trans0213);
    rope_info.add("is_chatglm", desc->config.is_chatglm);
    rope_info.add("support_2d_rope", desc->config.support_2d_rope);
    rope_info.add("output_trans0213", desc->config.output_trans0213);
    rope_info.add("is_interleaved", desc->config.is_interleaved);
    rope_info.add("is_qwen", desc->config.is_qwen);
    rope_info.add("rotary_ndims", desc->config.rotary_ndims);
    rope_info.add("slice_start", desc->config.slice_start);
    rope_info.add("slice_stop", desc->config.slice_stop);
    node_info->add("rope info", rope_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

}  // namespace cldnn
