// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "moe_gemm_inst.h"
#include "primitive_type_base.h"
#include "json_object.h"
#include "to_string_utils.h"
#include <string>
#include <vector>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(moe_gemm)

layout moe_gemm_inst::calc_output_layout(moe_gemm_node const& node, kernel_impl_params const& impl_param) {
    return calc_output_layouts<ov::PartialShape>(node, impl_param)[1];
}

template<typename ShapeType>
std::vector<layout> moe_gemm_inst::calc_output_layouts(moe_gemm_node const& /*node*/, const kernel_impl_params& impl_param) {
    const auto& desc = impl_param.typed_desc<moe_gemm>();
    size_t num_experts_per_token = desc->num_experts_per_token;
    auto input_layout = impl_param.get_input_layout(0);
    size_t input_rank = input_layout.get_partial_shape().size();
    if (input_rank != 3) {
        OPENVINO_THROW("moe_gemm's input rank should be 3");
    }
    auto experts_layout = impl_param.get_input_layout(1);
    auto out_n_dim = input_rank - 1;
    auto output_shape = input_layout.get_partial_shape();
    for (auto& o : output_shape) {
        o = ov::Dimension::dynamic();
    }
    size_t n = experts_layout.get_shape()[1];
    output_shape[out_n_dim] = ov::Dimension(n);

    if (!input_layout.is_dynamic()) {
        const auto& in_shape = input_layout.get_shape();
        const auto m = in_shape[0] * in_shape[1];
        // Follow the arrangement moe_gather produced; has_batch_dim only decides the ambiguous [1,1,K] case
        bool batch_layout = desc->has_batch_dim;
        if ((in_shape[0] == 1) != (in_shape[1] == 1)) {
            batch_layout = in_shape[0] == 1;
        }
        if (m == 1) {
            // first gemm (up/gate) in the generate phase
            if (!batch_layout) {
                output_shape = ov::PartialShape{ov::Dimension(num_experts_per_token), ov::Dimension(1), ov::Dimension(n)};
            } else {
                output_shape = ov::PartialShape{ov::Dimension(1), ov::Dimension(num_experts_per_token), ov::Dimension(n)};
            }
        } else {
            if (!batch_layout) {
                output_shape = ov::PartialShape{ov::Dimension(m), ov::Dimension(1), ov::Dimension(n)};
            } else {
                output_shape = ov::PartialShape{ov::Dimension(1), ov::Dimension(m), ov::Dimension(n)};
            }
        }
    }
    auto output_layout = layout{output_shape, input_layout.data_type, input_layout.format};
    return {output_layout};
}

template std::vector<layout> moe_gemm_inst::calc_output_layouts<ov::PartialShape>(moe_gemm_node const& node, const kernel_impl_params& impl_param);

std::string moe_gemm_inst::to_string(moe_gemm_node const& node) {
    auto node_info = node.desc_to_json();
    auto desc = node.get_primitive();

    std::stringstream primitive_description;

    json_composite moe_gemm_info;
    if (desc->output_data_types[0].has_value()) {
        moe_gemm_info.add("out dt: ", dt_to_str(*desc->output_data_types[0]));
    }
    node_info->dump(primitive_description);

    return primitive_description.str();
}

moe_gemm_inst::typed_primitive_inst(network& network, moe_gemm_node const& node) : parent(network, node) { }
}  // namespace cldnn
