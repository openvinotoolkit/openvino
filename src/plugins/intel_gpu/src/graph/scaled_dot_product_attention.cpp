// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "scaled_dot_product_attention_inst.h"

#include "primitive_type_base.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include <string>
#include <vector>

#include "scaled_dot_product_attention_shape_inference.hpp"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(scaled_dot_product_attention)

layout scaled_dot_product_attention_inst::calc_output_layout(scaled_dot_product_attention_node const& /* node */,
                                                             kernel_impl_params const& impl_param) {
    auto desc = impl_param.typed_desc<scaled_dot_product_attention>();

    return impl_param.get_input_layout(0);
}

template<typename ShapeType>
std::vector<layout> scaled_dot_product_attention_inst::calc_output_layouts(scaled_dot_product_attention_node const& /*node*/,
                                                                           const kernel_impl_params& impl_param) {
    return { impl_param.get_input_layout(0) };
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
    scaled_dot_product_attention_info.add("has_attn_mask_input", desc->has_attn_mask_input);
    scaled_dot_product_attention_info.add("has_scale_input", desc->has_scale_input);

    node_info->add("scaled_dot_product_attention_info", scaled_dot_product_attention_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

scaled_dot_product_attention_inst::typed_primitive_inst(network& network, scaled_dot_product_attention_node const& node)
    : parent(network, node) {}

}  // namespace cldnn
