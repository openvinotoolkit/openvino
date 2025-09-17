// Copyright (C) 2025 Intel Corporation
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
    // TODO
    std::cout << "static shape!!! calc_output_layout" << std::endl;
    return impl_param.input_layouts[0];
}

template<typename ShapeType>
std::vector<layout> moe_gemm_inst::calc_output_layouts(moe_gemm_node const& /*node*/, const kernel_impl_params& impl_param) {
    auto input_layout = impl_param.get_input_layout(0);
    auto experts_layout = impl_param.get_input_layout(1); // [experts, N, K]

//    size_t total_experts = experts_layout.get_shape()[0];
//    auto m = input_layout.get_partial_shape()[1]; // [experts, seq_len, K]
//    size_t n = experts_layout.get_shape()[1];
    size_t n = experts_layout.get_shape()[1];

//    ov::PartialShape output_shape = { ov::Dimension(total_experts), m, ov::Dimension(n) };
    ov::PartialShape output_shape;
    if (input_layout.is_dynamic()) {
        output_shape = { ov::Dimension::dynamic(), ov::Dimension(n) };
    } else {
        auto m = input_layout.get_shape()[0]; // [num_actual_experts * seq_len, K]
        if (m == 1) // first gemm in the generate phase
            output_shape = {ov::Dimension(impl_param.get_input_layout(3).get_shape()[0]), ov::Dimension(n)};
        else
            output_shape = { ov::Dimension(m), ov::Dimension(n) };
    }
    std::cout << "calc_output_layouts" << std::endl;
    auto output_layout = layout{ output_shape, data_types::f16, format::bfyx };
    std::cout << output_layout.to_short_string() << std::endl;
    return {output_layout};
}

template std::vector<layout> moe_gemm_inst::calc_output_layouts<ov::PartialShape>(moe_gemm_node const& node, const kernel_impl_params& impl_param);

std::string moe_gemm_inst::to_string(moe_gemm_node const& node) {
    auto node_info = node.desc_to_json();
    auto desc = node.get_primitive();

    std::stringstream primitive_description;

    json_composite moe_gemm_info;
    if (desc->output_data_types[0].has_value())
        moe_gemm_info.add("out dt: ", dt_to_str(*desc->output_data_types[0]));
    node_info->dump(primitive_description);

    return primitive_description.str();
}

moe_gemm_inst::typed_primitive_inst(network& network, moe_gemm_node const& node) : parent(network, node) { }
}  // namespace cldnn
