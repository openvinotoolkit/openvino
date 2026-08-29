// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>

#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include "moe_router_fused_inst.h"
#include "openvino/core/except.hpp"
#include "primitive_type_base.h"
#include "program_node.h"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(moe_router_fused)

layout moe_router_fused_inst::calc_output_layout(const moe_router_fused_node& /* node */, const kernel_impl_params& impl_param) {
    auto desc = impl_param.typed_desc<moe_router_fused>();
    auto input_layout = impl_param.get_input_layout(0);
    auto shape = input_layout.get_shape();
    size_t num_tokens = shape[0];
    if (shape.size() == 3)
        num_tokens = shape[0] * shape[1];
    size_t top_k = desc->_config.top_k;
    return layout(ov::Shape{num_tokens, top_k}, input_layout.data_type, format::bfyx);
}

template <typename ShapeType>
std::vector<layout> moe_router_fused_inst::calc_output_layouts(const moe_router_fused_node& /* node */, const kernel_impl_params& impl_param) {
    auto desc = impl_param.typed_desc<moe_router_fused>();
    auto input_layout = impl_param.get_input_layout(0);
    auto input_pshape = input_layout.get_partial_shape();
    size_t top_k = desc->_config.top_k;

    OPENVINO_ASSERT(input_pshape.rank().is_static(), "Input rank must be static");
    ov::PartialShape out_shape;
    auto num_tokens = input_pshape[0];
    if (input_pshape.rank().get_length() == 3) {
        // 3D input [batch, seq_len, num_experts] -> num_tokens = batch * seq_len
        num_tokens = input_pshape[0] * input_pshape[1];
    }
    out_shape = ov::PartialShape{num_tokens, static_cast<int64_t>(top_k)};

    // Output 0: topk_weights [num_tokens, top_k] — same element type as input
    layout weights_layout(out_shape, input_layout.data_type, format::bfyx);
    // Output 1: topk_indices [num_tokens, top_k] — i32
    layout indices_layout(out_shape, ov::element::i32, format::bfyx);

    return {weights_layout, indices_layout};
}

template std::vector<layout> moe_router_fused_inst::calc_output_layouts<ov::PartialShape>(const moe_router_fused_node& node, const kernel_impl_params& impl_param);

std::string moe_router_fused_inst::to_string(const moe_router_fused_node& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    json_composite info;
    info.add("num_expert", desc->_config.num_expert);
    info.add("top_k", desc->_config.top_k);
    info.add("routing_type", static_cast<int>(desc->_config.routing_type));
    node_info->add("moe_router_fused info", info);

    std::stringstream primitive_description;
    node_info->dump(primitive_description);
    return primitive_description.str();
}

moe_router_fused_inst::typed_primitive_inst(network& network, const moe_router_fused_node& node) : parent(network, node) {}

}  // namespace cldnn
