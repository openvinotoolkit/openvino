// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "grouped_matmul_inst.h"
#include "primitive_type_base.h"
#include "json_object.h"
#include "to_string_utils.h"

#include <string>
#include <vector>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(grouped_matmul)

layout grouped_matmul_inst::calc_output_layout(grouped_matmul_node const& node, kernel_impl_params const& impl_param) {
    return calc_output_layouts<ov::PartialShape>(node, impl_param)[0];
}

template <typename ShapeType>
std::vector<layout> grouped_matmul_inst::calc_output_layouts(grouped_matmul_node const& /*node*/,
                                                             const kernel_impl_params& impl_param) {
    // mat_a: [total_tokens, K], mat_b: [G, N, K], out: [total_tokens, N]
    auto input_layout = impl_param.get_input_layout(grouped_matmul::GroupedMatmulInputIdx::INPUT);
    auto weight_layout = impl_param.get_input_layout(grouped_matmul::GroupedMatmulInputIdx::WEIGHT);

    const auto& a_pshape = input_layout.get_partial_shape();
    const auto& b_pshape = weight_layout.get_partial_shape();

    OPENVINO_ASSERT(a_pshape.rank().is_static() && a_pshape.size() == 2,
                    "grouped_matmul mat_a rank must be static and equal to 2, got ",
                    a_pshape);
    OPENVINO_ASSERT(b_pshape.rank().is_static() && b_pshape.size() == 3,
                    "grouped_matmul mat_b rank must be static and equal to 3, got ",
                    b_pshape);

    const auto desc = impl_param.typed_desc<grouped_matmul>();
    const auto output_type = desc->output_data_types[0].value_or(input_layout.data_type);

    ov::PartialShape output_pshape{a_pshape[0], b_pshape[1]};
    return {layout{output_pshape, output_type, input_layout.format}};
}

template std::vector<layout> grouped_matmul_inst::calc_output_layouts<ov::PartialShape>(
    grouped_matmul_node const& node,
    const kernel_impl_params& impl_param);

std::string grouped_matmul_inst::to_string(grouped_matmul_node const& node) {
    auto node_info = node.desc_to_json();
    auto desc = node.get_primitive();

    std::stringstream primitive_description;

    json_composite info;
    if (desc->output_data_types[0].has_value())
        info.add("out dt: ", dt_to_str(*desc->output_data_types[0]));
    node_info->add("grouped_matmul_info", info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

grouped_matmul_inst::typed_primitive_inst(network& network, grouped_matmul_node const& node) : parent(network, node) {}
}  // namespace cldnn
