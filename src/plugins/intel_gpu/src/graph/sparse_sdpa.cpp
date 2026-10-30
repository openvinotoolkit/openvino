// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <sstream>
#include <string>
#include <vector>

#include "json_object.h"
#include "primitive_type_base.h"
#include "sparse_sdpa_inst.h"
#include "to_string_utils.h"

namespace cldnn {

GPU_DEFINE_PRIMITIVE_TYPE_ID(sparse_sdpa)

layout sparse_sdpa_inst::calc_output_layout(const sparse_sdpa_node& node, const kernel_impl_params& impl_param) {
    return calc_output_layouts<ov::PartialShape>(node, impl_param)[0];
}

template <typename ShapeType>
std::vector<layout> sparse_sdpa_inst::calc_output_layouts(const sparse_sdpa_node& node, const kernel_impl_params& impl_param) {
    const auto& desc = impl_param.typed_desc<sparse_sdpa>();
    const auto query_layout = impl_param.get_input_layout(sparse_sdpa::InputIdx::QUERY);
    auto query_shape = query_layout.get<ShapeType>();
    OPENVINO_ASSERT(query_shape.rank().is_static() && query_shape.size() == 4, "[GPU] sparse_sdpa expects a rank-4 [B, Hq, L, Dh] query input");

    ShapeType out_shape{query_shape[0],
                        ov::Dimension(static_cast<int64_t>(desc->num_heads)),
                        query_shape[2],
                        ov::Dimension(static_cast<int64_t>(desc->v_head_size))};

    const auto out_dt =
        desc->output_data_types.empty() || !desc->output_data_types[0].has_value() ? query_layout.data_type : desc->output_data_types[0].value();
    return {layout{out_shape, out_dt, query_layout.format}};
}

template std::vector<layout> sparse_sdpa_inst::calc_output_layouts<ov::PartialShape>(const sparse_sdpa_node& node, const kernel_impl_params& impl_param);

std::string sparse_sdpa_inst::to_string(const sparse_sdpa_node& node) {
    auto node_info = node.desc_to_json();
    auto desc = node.get_primitive();

    std::stringstream primitive_description;
    json_composite info;
    info.add("num_heads", desc->num_heads);
    info.add("num_kv_heads", desc->num_kv_heads);
    info.add("num_selection_heads", desc->num_selection_heads);
    info.add("k_head_size", desc->k_head_size);
    info.add("v_head_size", desc->v_head_size);
    info.add("key_block_size", desc->key_block_size);
    info.add("causal", desc->causal);
    info.add("scale", desc->scale);
    node_info->add("sparse_sdpa_info", info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

sparse_sdpa_inst::typed_primitive_inst(network& network, const sparse_sdpa_node& node) : parent(network, node) {
    std::string reason;
    OPENVINO_ASSERT(ov::intel_gpu::op::SparseSDPA::validate_config(node.get_primitive()->get_config(), reason),
                    "[GPU] Invalid sparse_sdpa primitive: ",
                    reason);
}

}  // namespace cldnn
