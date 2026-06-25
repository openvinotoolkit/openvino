// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>

#include "grouped_matmul_inst.h"
#include "json_object.h"
#include "primitive_type_base.h"
#include "to_string_utils.h"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(grouped_matmul)

layout grouped_matmul_inst::calc_output_layout(const grouped_matmul_node& node, const kernel_impl_params& impl_param) {
    return calc_output_layouts<ov::PartialShape>(node, impl_param)[0];
}

template <typename ShapeType>
std::vector<layout> grouped_matmul_inst::calc_output_layouts(const grouped_matmul_node& /*node*/, const kernel_impl_params& impl_param) {
    const auto& desc = impl_param.typed_desc<grouped_matmul>();
    const auto& input_layout = impl_param.get_input_layout(grouped_matmul::InputIdx::INPUT);
    const auto& weight_layout = impl_param.get_input_layout(grouped_matmul::InputIdx::WEIGHT);

    const auto a_shape = input_layout.get_partial_shape();
    const auto b_shape = weight_layout.get_partial_shape();

    // Use has_offsets from the primitive (set at program-builder time from the OV op)
    // rather than checking shape ranks, because the cldnn layout system may pad shapes to 4D.
    ov::PartialShape output_shape;
    if (!desc->has_offsets) {
        // 3D×3D: A:[G,M,K], B:[G,N,K] -> out:[G,M,N]
        // Logical dims: dim[-3]=G, dim[-2]=M, dim[-1]=K for A; dim[-2]=N for B.
        OPENVINO_ASSERT(b_shape.rank().is_static() && b_shape.size() >= 2,
                        "grouped_matmul 3D×3D: B must have at least rank 2");
        OPENVINO_ASSERT(a_shape.rank().is_static() && a_shape.size() >= 3,
                        "grouped_matmul 3D×3D: A must have at least rank 3");
        const ov::Dimension G = a_shape[a_shape.size() - 3];
        const ov::Dimension M = a_shape[a_shape.size() - 2];
        const ov::Dimension N = b_shape[b_shape.size() - 2];
        output_shape = {G, M, N};
    } else {
        // 2D×3D: A:[T,K], B:[G,N,K], offsets:[G] -> out:[T,N]
        OPENVINO_ASSERT(a_shape.rank().is_static() && a_shape.size() >= 2,
                        "grouped_matmul 2D×3D: A must have at least rank 2");
        OPENVINO_ASSERT(b_shape.rank().is_static() && b_shape.size() >= 2,
                        "grouped_matmul 2D×3D: B must have at least rank 2");
        const ov::Dimension T = a_shape[a_shape.size() - 2];
        const ov::Dimension N = b_shape[b_shape.size() - 2];
        output_shape = {T, N};
    }

    return {layout{output_shape, input_layout.data_type, input_layout.format}};
}

template std::vector<layout> grouped_matmul_inst::calc_output_layouts<ov::PartialShape>(const grouped_matmul_node& node, const kernel_impl_params& impl_param);

std::string grouped_matmul_inst::to_string(const grouped_matmul_node& node) {
    auto node_info = node.desc_to_json();
    auto desc = node.get_primitive();

    std::stringstream primitive_description;

    json_composite info;
    info.add("has_offsets", desc->has_offsets);
    info.add("has_bias", desc->has_bias);
    node_info->add("grouped_matmul_info", info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

grouped_matmul_inst::typed_primitive_inst(network& network, const grouped_matmul_node& node) : parent(network, node) {}

}  // namespace cldnn
