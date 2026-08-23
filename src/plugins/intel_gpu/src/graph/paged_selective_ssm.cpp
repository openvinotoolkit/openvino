// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/paged_selective_ssm.hpp"

#include <string>
#include <vector>

#include "json_object.h"
#include "paged_selective_ssm_inst.h"
#include "paged_selective_ssm_shape_inference.hpp"
#include "primitive_type_base.h"
#include "to_string_utils.h"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(paged_selective_ssm)

layout paged_selective_ssm_inst::calc_output_layout(const paged_selective_ssm_node& node, const kernel_impl_params& impl_param) {
    return calc_output_layouts<ov::PartialShape>(node, impl_param)[0];
}

template <typename ShapeType>
std::vector<layout> paged_selective_ssm_inst::calc_output_layouts(const paged_selective_ssm_node& node, const kernel_impl_params& impl_param) {
    const auto& all_inputs = node.get_input_layouts();
    OPENVINO_ASSERT(all_inputs.size() == 11, "paged_selective_ssm must have 11 inputs");

    std::vector<ShapeType> input_shapes;
    input_shapes.reserve(all_inputs.size());
    for (size_t i = 0; i < all_inputs.size(); i++) {
        input_shapes.push_back(impl_param.get_input_layout(i).get<ShapeType>());
    }

    ov::op::internal::PagedSelectiveSSM op;
    const auto output_shapes = ov::op::internal::shape_infer(&op, input_shapes);
    const auto x_layout = impl_param.get_input_layout(3);
    return {layout(output_shapes[0], x_layout.data_type, x_layout.format)};
}

template std::vector<layout> paged_selective_ssm_inst::calc_output_layouts<ov::PartialShape>(const paged_selective_ssm_node& node,
                                                                                             const kernel_impl_params& impl_param);

std::string paged_selective_ssm_inst::to_string(const paged_selective_ssm_node& node) {
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;
    json_composite pssm_info;
    pssm_info.add("A", node.input(0).id());
    pssm_info.add("dt", node.input(1).id());
    pssm_info.add("B", node.input(2).id());
    pssm_info.add("x", node.input(3).id());
    pssm_info.add("C", node.input(4).id());
    pssm_info.add("recurrent_state_table", node.input(5).id());
    pssm_info.add("subsequence_begins", node.input(6).id());
    pssm_info.add("block_indices", node.input(7).id());
    pssm_info.add("block_indices_begins", node.input(8).id());
    pssm_info.add("num_processed_tokens", node.input(9).id());
    pssm_info.add("cache_interval", node.input(10).id());
    node_info->add("paged_selective_ssm_info", pssm_info);
    node_info->dump(primitive_description);
    return primitive_description.str();
}

paged_selective_ssm_inst::typed_primitive_inst(network& network, const paged_selective_ssm_node& node) : parent(network, node) {}

}  // namespace cldnn
