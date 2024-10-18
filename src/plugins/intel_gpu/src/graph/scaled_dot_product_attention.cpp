// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "scaled_dot_product_attention_inst.h"

#include "primitive_type_base.h"
#include "json_object.h"
#include <string>
#include <vector>

#include "scaled_dot_product_attention_shape_inference.hpp"
#include "intel_gpu/op/sdpa.hpp"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(scaled_dot_product_attention)

layout scaled_dot_product_attention_inst::calc_output_layout(scaled_dot_product_attention_node const& /* node */,
                                                             kernel_impl_params const& impl_param) {
    auto desc = impl_param.typed_desc<scaled_dot_product_attention>();

    auto transpose_shape = [&desc](const ov::PartialShape& shape, const std::vector<int64_t>& order) {
        if (desc->input_q_transpose_order.empty())
            return shape;

        auto shape_transposed = ov::PartialShape(shape);
        auto rank_diff = shape.size() - order.size();
        for (size_t i = 0; i < order.size(); i++) {
            size_t idx = static_cast<size_t>(order[i]);
            shape_transposed[i + rank_diff] = shape[idx + rank_diff];
        }

        return shape_transposed;
    };

    auto input0_layout = impl_param.get_input_layout(0);
    auto default_out_dt = data_type_traits::is_floating_point(input0_layout.data_type) ? input0_layout.data_type : data_types::f32;
    auto output_type = desc->output_data_types[0].value_or(default_out_dt);
    auto output_format = input0_layout.format;
    auto output_shape = transpose_shape(input0_layout.get_partial_shape(), desc->input_q_transpose_order); // output shape matches with Q input shape

    return { layout{output_shape, output_type, output_format, desc->output_paddings[0]} };
}

template<typename ShapeType>
std::vector<layout> scaled_dot_product_attention_inst::calc_output_layouts(scaled_dot_product_attention_node const& /*node*/,
                                                                           const kernel_impl_params& impl_param) {
    auto prim = impl_param.typed_desc<scaled_dot_product_attention>();
    const auto& input0_layout = impl_param.get_input_layout(0);

    auto default_out_dt = data_type_traits::is_floating_point(input0_layout.data_type) ? input0_layout.data_type : data_types::f32;
    auto output_type = prim->output_data_types[0].value_or(default_out_dt);

    if (impl_param.has_fused_primitives()) {
        output_type = impl_param.get_output_element_type();
    }

    ov::intel_gpu::op::SDPA op;

    std::vector<ShapeType> input_shapes;
    for (size_t i = 0; i < impl_param.input_layouts.size(); i++) {
        input_shapes.push_back(impl_param.get_input_layout(0).get<ShapeType>());
    }

    std::vector<ShapeType> output_shapes = ov::intel_gpu::op::shape_infer(&op,
                                                                          input_shapes,
                                                                          prim->input_q_transpose_order,
                                                                          prim->input_k_transpose_order,
                                                                          prim->input_v_transpose_order,
                                                                          prim->output_transpose_order);

    cldnn::format output_format = input0_layout.format;

    return { layout{output_shapes[0], output_type, output_format, prim->output_paddings[0]} };
}

template std::vector<layout> scaled_dot_product_attention_inst::calc_output_layouts<ov::PartialShape>(scaled_dot_product_attention_node const& node,
                                                                                                      const kernel_impl_params& impl_param);

std::string scaled_dot_product_attention_inst::to_string(scaled_dot_product_attention_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto& input = node.input();

    std::stringstream primitive_description;

    json_composite scaled_dot_product_attention_info;
    scaled_dot_product_attention_info.add("input id", input.id());
    scaled_dot_product_attention_info.add("is_causal", desc->is_causal);
    scaled_dot_product_attention_info.add("is_kv_compressed", desc->is_kv_compressed);
    scaled_dot_product_attention_info.add("combine_scales_and_zp", desc->combine_scales_and_zp);
    scaled_dot_product_attention_info.add("group_size", desc->quantization_config.group_sizes);
    scaled_dot_product_attention_info.add("is_asymmetric_quantization", desc->quantization_config.is_asymmetric_quantization());
    scaled_dot_product_attention_info.add("quantization_dt", desc->quantization_config.quantization_dt);
    scaled_dot_product_attention_info.add("scale_dt", desc->quantization_config.scale_dt);
    scaled_dot_product_attention_info.add("zp_dt", desc->quantization_config.zp_dt);
    scaled_dot_product_attention_info.add("indirect_axis", desc->indirect_axis);
    scaled_dot_product_attention_info.add("has_attn_mask_input", desc->has_attn_mask_input);
    scaled_dot_product_attention_info.add("has_scale_input", desc->has_scale_input);
    scaled_dot_product_attention_info.add("input_q_transpose_order", desc->input_q_transpose_order);
    scaled_dot_product_attention_info.add("input_k_transpose_order", desc->input_k_transpose_order);
    scaled_dot_product_attention_info.add("input_v_transpose_order", desc->input_v_transpose_order);
    scaled_dot_product_attention_info.add("output_transpose_order", desc->output_transpose_order);

    node_info->add("scaled_dot_product_attention_info", scaled_dot_product_attention_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

scaled_dot_product_attention_inst::typed_primitive_inst(network& network, scaled_dot_product_attention_node const& node)
    : parent(network, node) {}

}  // namespace cldnn
