// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>

#include "json_object.h"
#include "primitive_type_base.h"
#include "qsa_inst.h"
#include "to_string_utils.h"

namespace cldnn {

GPU_DEFINE_PRIMITIVE_TYPE_ID(qsa)

layout qsa_inst::calc_output_layout(const qsa_node& node, const kernel_impl_params& impl_param) {
    return calc_output_layouts<ov::PartialShape>(node, impl_param)[0];
}

template <typename ShapeType>
std::vector<layout> qsa_inst::calc_output_layouts(const qsa_node& /*node*/, const kernel_impl_params& impl_param) {
    const auto& desc = impl_param.typed_desc<qsa>();
    const auto query_layout = impl_param.get_input_layout(qsa::QSAInputIdx::QUERY);

    auto out_shape = ShapeType{query_layout.get<ShapeType>()};
    OPENVINO_ASSERT(out_shape.rank().is_static() && out_shape.rank().get_length() == 2, "[GPU] qsa expects a rank-2 [tokens, Hq*Dh] query input");
    out_shape[1] = ov::Dimension(static_cast<int64_t>(desc->heads_num * desc->v_head_size));

    const auto out_dt =
        desc->output_data_types.empty() || !desc->output_data_types[0].has_value() ? query_layout.data_type : desc->output_data_types[0].value();

    return {layout{out_shape, out_dt, query_layout.format}};
}

template std::vector<layout> qsa_inst::calc_output_layouts<ov::PartialShape>(const qsa_node& node, const kernel_impl_params& impl_param);

std::string qsa_inst::to_string(const qsa_node& node) {
    auto node_info = node.desc_to_json();
    auto desc = node.get_primitive();

    std::stringstream primitive_description;

    json_composite info;
    info.add("heads_num", desc->heads_num);
    info.add("kv_heads_num", desc->kv_heads_num);
    info.add("k_head_size", desc->k_head_size);
    info.add("v_head_size", desc->v_head_size);
    info.add("hidden_size", desc->hidden_size);
    info.add("indexer_heads", desc->indexer_heads);
    info.add("indexer_kv_heads", desc->indexer_kv_heads);
    info.add("indexer_head_dim", desc->indexer_head_dim);
    info.add("indexer_rotary_dim", desc->indexer_rotary_dim);
    info.add("budget", desc->budget);
    info.add("compress_ratio", desc->compress_ratio);
    info.add("qsa_block_tokens", desc->qsa_block_tokens);
    info.add("pa_block_size", desc->pa_block_size);
    info.add("block_topk", desc->block_topk);
    info.add("scale", desc->scale);
    info.add("has_output_gate", desc->has_output_gate);
    info.add("is_causal", desc->is_causal);

    node_info->add("qsa_info", info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

qsa_inst::typed_primitive_inst(network& network, const qsa_node& node) : parent(network, node) {
    std::string reason;
    OPENVINO_ASSERT(ov::intel_gpu::op::QSA::validate_config(node.get_primitive()->get_config(), reason), "[GPU] Invalid qsa primitive: ", reason);
}

}  // namespace cldnn
