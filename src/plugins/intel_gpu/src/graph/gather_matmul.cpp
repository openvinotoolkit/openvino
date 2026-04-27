// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>

#include "gather_matmul_inst.h"
#include "intel_gpu/primitives/swiglu.hpp"
#include "json_object.h"
#include "primitive_type_base.h"
#include "to_string_utils.h"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(gather_matmul)

layout gather_matmul_inst::calc_output_layout(const gather_matmul_node& node, const kernel_impl_params& impl_param) {
    return calc_output_layouts<ov::PartialShape>(node, impl_param)[0];
}

template <typename ShapeType>
std::vector<layout> gather_matmul_inst::calc_output_layouts(const gather_matmul_node& /*node*/, const kernel_impl_params& impl_param) {
    // A: [n_activated_experts, batch*seq, hidden_size]
    // B: [n_all_experts, N, K] (transposed weights)
    // indices: [batch*seq, top_k]
    // Output: [top_k, batch*seq, N]
    auto input_layout = impl_param.get_input_layout(gather_matmul::BGMInputIdx::INPUT);
    auto weight_layout = impl_param.get_input_layout(gather_matmul::BGMInputIdx::WEIGHT);
    auto indices_layout = impl_param.get_input_layout(gather_matmul::BGMInputIdx::INDICES);

    size_t input_rank = input_layout.get_partial_shape().size();
    OPENVINO_ASSERT(input_rank == 3, "gather_matmul input rank should be 3, got ", input_rank);

    auto a_shape = input_layout.get_partial_shape();
    auto weight_shape = weight_layout.get_partial_shape();
    auto indices_shape = indices_layout.get_partial_shape();

    // Output dim[0] = top_k (from indices dim[1]), not n_activated_experts (from A dim[0])
    ov::Dimension top_k = (indices_shape.rank().is_static() && indices_shape.rank().get_length() >= 2) ? indices_shape[1] : ov::Dimension::dynamic();
    ov::Dimension n_tokens = a_shape[1];
    ov::Dimension out_features = (weight_shape.rank().is_static() && weight_shape.rank().get_length() >= 2) ? weight_shape[1] : ov::Dimension::dynamic();

    // When SwiGLU is fused, output features are halved (weights have 2N features: gate + value)
    for (const auto& fd : impl_param.fused_desc) {
        if (fd.is_type<swiglu>()) {
            if (out_features.is_static())
                out_features = out_features.get_length() / 2;
            else
                out_features = ov::Dimension::dynamic();
            break;
        }
    }

    ov::PartialShape output_shape = {top_k, n_tokens, out_features};
    auto output_layout = layout{output_shape, input_layout.data_type, input_layout.format};
    return {output_layout};
}

template std::vector<layout> gather_matmul_inst::calc_output_layouts<ov::PartialShape>(const gather_matmul_node& node, const kernel_impl_params& impl_param);

std::string gather_matmul_inst::to_string(const gather_matmul_node& node) {
    auto node_info = node.desc_to_json();
    auto desc = node.get_primitive();

    std::stringstream primitive_description;

    json_composite bgm_info;
    bgm_info.add("has_bias", desc->has_bias);
    bgm_info.add("has_zp", desc->has_zp);
    bgm_info.add("n_activated_experts", desc->n_activated_experts);
    if (desc->output_data_types[0].has_value())
        bgm_info.add("out dt: ", dt_to_str(*desc->output_data_types[0]));
    node_info->add("gather_matmul_info", bgm_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

gather_matmul_inst::typed_primitive_inst(network& network, const gather_matmul_node& node) : parent(network, node) {}
}  // namespace cldnn
