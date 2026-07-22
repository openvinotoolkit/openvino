// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/paged_gated_delta_net.hpp"

#include <string>
#include <vector>

#include "json_object.h"
#include "paged_gated_delta_net_inst.h"
#include "paged_gated_delta_net_shape_inference.hpp"
#include "primitive_type_base.h"
#include "to_string_utils.h"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(paged_gated_delta_net)

layout paged_gated_delta_net_inst::calc_output_layout(const paged_gated_delta_net_node& node, const kernel_impl_params& impl_param) {
    return calc_output_layouts<ov::PartialShape>(node, impl_param)[0];
}

template <typename ShapeType>
std::vector<layout> paged_gated_delta_net_inst::calc_output_layouts(const paged_gated_delta_net_node& node, const kernel_impl_params& impl_param) {
    const auto& all_inputs = node.get_input_layouts();
    OPENVINO_ASSERT(all_inputs.size() == 11, "paged_gated_delta_net must have 11 inputs");

    const auto value_layout = impl_param.get_input_layout(2);

    std::vector<ShapeType> input_shapes;
    input_shapes.reserve(all_inputs.size());
    for (size_t i = 0; i < all_inputs.size(); i++) {
        input_shapes.push_back(impl_param.get_input_layout(i).get<ShapeType>());
    }

    ov::op::internal::PagedGatedDeltaNet op;
    const auto output_shapes = ov::op::internal::shape_infer(&op, input_shapes);

    return {layout(output_shapes[0], value_layout.data_type, value_layout.format)};
}

template std::vector<layout> paged_gated_delta_net_inst::calc_output_layouts<ov::PartialShape>(const paged_gated_delta_net_node& node,
                                                                                               const kernel_impl_params& impl_param);

std::string paged_gated_delta_net_inst::to_string(const paged_gated_delta_net_node& node) {
    auto node_info = node.desc_to_json();
    auto desc = node.get_primitive();

    std::stringstream primitive_description;

    json_composite paged_gdn_info;
    paged_gdn_info.add("query", node.input(0).id());
    paged_gdn_info.add("key", node.input(1).id());
    paged_gdn_info.add("value", node.input(2).id());
    paged_gdn_info.add("recurrent_state_table", node.input(3).id());
    paged_gdn_info.add("gate", node.input(4).id());
    paged_gdn_info.add("beta", node.input(5).id());
    paged_gdn_info.add("subsequence_begins", node.input(6).id());
    paged_gdn_info.add("block_indices", node.input(7).id());
    paged_gdn_info.add("block_indices_begins", node.input(8).id());
    paged_gdn_info.add("past_lens", node.input(9).id());
    paged_gdn_info.add("cache_interval", node.input(10).id());
    paged_gdn_info.add("k_head_size", desc->k_head_size);
    paged_gdn_info.add("v_head_size", desc->v_head_size);
    paged_gdn_info.add("k_heads_num", desc->k_heads_num);
    paged_gdn_info.add("v_heads_num", desc->v_heads_num);

    node_info->add("paged_gated_delta_net_info", paged_gdn_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

paged_gated_delta_net_inst::typed_primitive_inst(network& network, const paged_gated_delta_net_node& node) : parent(network, node) {}

}  // namespace cldnn
