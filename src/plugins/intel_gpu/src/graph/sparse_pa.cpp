// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

#include "json_object.h"
#include "primitive_type_base.h"
#include "sparse_pa_inst.h"
#include "to_string_utils.h"

namespace cldnn {

GPU_DEFINE_PRIMITIVE_TYPE_ID(sparse_pa)

layout sparse_pa_inst::calc_output_layout(const sparse_pa_node& node, const kernel_impl_params& impl_param) {
    return calc_output_layouts<ov::PartialShape>(node, impl_param)[0];
}

template <typename ShapeType>
std::vector<layout> sparse_pa_inst::calc_output_layouts(const sparse_pa_node& node, const kernel_impl_params& impl_param) {
    const auto& desc = impl_param.typed_desc<sparse_pa>();
    const auto query_layout = impl_param.get_input_layout(sparse_pa::InputIdx::QUERY);

    auto out_shape = ShapeType{query_layout.get<ShapeType>()};
    OPENVINO_ASSERT(out_shape.rank().is_static() && out_shape.size() >= 2 &&
                        std::all_of(out_shape.begin() + 2, out_shape.end(), [](const ov::Dimension& dim) { return dim.compatible(1); }),
                    "[GPU] sparse_pa expects a rank-2 [tokens, num_heads*k_head_size] query input, got ",
                    query_layout.to_short_string());
    out_shape[1] = ov::Dimension(static_cast<int64_t>(desc->num_heads * desc->v_head_size));

    const auto out_dt = desc->output_data_types.empty() || !desc->output_data_types[0].has_value() ? query_layout.data_type : desc->output_data_types[0].value();
    return {layout{out_shape, out_dt, query_layout.format}};
}

template std::vector<layout> sparse_pa_inst::calc_output_layouts<ov::PartialShape>(const sparse_pa_node& node, const kernel_impl_params& impl_param);

std::string sparse_pa_inst::to_string(const sparse_pa_node& node) {
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
    info.add("scale", desc->scale);
    info.add("causal", desc->causal);
    node_info->add("sparse_pa_info", info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

sparse_pa_inst::typed_primitive_inst(network& network, const sparse_pa_node& node) : parent(network, node) {
    std::string reason;
    OPENVINO_ASSERT(sparse_pa::validate_config(node.get_primitive()->get_config(), reason), "[GPU] Invalid sparse_pa primitive: ", reason);
}

}  // namespace cldnn
