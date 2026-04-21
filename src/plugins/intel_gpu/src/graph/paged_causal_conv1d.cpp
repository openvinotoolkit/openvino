// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/paged_causal_conv1d.hpp"

#include <string>
#include <vector>

#include "json_object.h"
#include "paged_causal_conv1d_inst.h"
#include "paged_causal_conv1d_shape_inference.hpp"
#include "primitive_type_base.h"
#include "to_string_utils.h"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(paged_causal_conv1d)

layout paged_causal_conv1d_inst::calc_output_layout(const paged_causal_conv1d_node& node, const kernel_impl_params& impl_param) {
    return calc_output_layouts<ov::PartialShape>(node, impl_param)[0];
}

template <typename ShapeType>
std::vector<layout> paged_causal_conv1d_inst::calc_output_layouts(const paged_causal_conv1d_node& node, const kernel_impl_params& impl_param) {
    const auto& all_inputs = node.get_input_layouts();
    OPENVINO_ASSERT(all_inputs.size() == 9, "paged_causal_conv1d must have 9 inputs");

    const auto input_layout = impl_param.get_input_layout(0);

    std::vector<ShapeType> input_shapes;
    input_shapes.reserve(all_inputs.size());
    for (size_t i = 0; i < all_inputs.size(); i++) {
        input_shapes.push_back(impl_param.get_input_layout(i).get<ShapeType>());
    }

    ov::op::internal::PagedCausalConv1D op;
    const auto output_shapes = ov::op::internal::shape_infer(&op, input_shapes);

    return {layout(output_shapes[0], input_layout.data_type, input_layout.format)};
}

template std::vector<layout> paged_causal_conv1d_inst::calc_output_layouts<ov::PartialShape>(const paged_causal_conv1d_node& node,
                                                                                             const kernel_impl_params& impl_param);

std::string paged_causal_conv1d_inst::to_string(const paged_causal_conv1d_node& node) {
    auto node_info = node.desc_to_json();
    auto desc = node.get_primitive();

    std::stringstream primitive_description;

    json_composite pcc_info;
    pcc_info.add("input_embeds", node.input(0).id());
    pcc_info.add("conv_state_table", node.input(1).id());
    pcc_info.add("conv_weight", node.input(2).id());
    pcc_info.add("conv_bias", node.input(3).id());
    pcc_info.add("subsequence_begins", node.input(4).id());
    pcc_info.add("block_indices", node.input(5).id());
    pcc_info.add("block_indices_begins", node.input(6).id());
    pcc_info.add("past_lens", node.input(7).id());
    pcc_info.add("cache_interval", node.input(8).id());
    pcc_info.add("hidden_size", desc->hidden_size);
    pcc_info.add("kernel_size", desc->kernel_size);

    node_info->add("paged_causal_conv1d_info", pcc_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

paged_causal_conv1d_inst::typed_primitive_inst(network& network, const paged_causal_conv1d_node& node) : parent(network, node) {}

}  // namespace cldnn
