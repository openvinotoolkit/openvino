// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "json_object.h"
#include "msda_inst.h"
#include "primitive_type_base.h"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(msda);

template <typename ShapeType>
std::vector<layout> msda_inst::calc_output_layouts(const msda_node& /*node*/, const kernel_impl_params& impl_param) {
    // value [B, S, H, D] and attention_weights [B, Q, H, L, P] give the output [B, Q, H * D].
    const auto& value_layout = impl_param.get_input_layout(0);
    const auto value = value_layout.get_partial_shape();
    const auto weights = impl_param.get_input_layout(4).get_partial_shape();
    const ov::PartialShape output_shape{value[0], weights[1], value[2] * value[3]};
    return {layout{output_shape, value_layout.data_type, format::get_default_format(output_shape.size())}};
}

template std::vector<layout> msda_inst::calc_output_layouts<ov::PartialShape>(const msda_node& node, const kernel_impl_params& impl_param);

std::string msda_inst::to_string(const msda_node& node) {
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;
    node_info->dump(primitive_description);

    return primitive_description.str();
}

msda_inst::typed_primitive_inst(network& network, const msda_node& node) : parent(network, node) {}

}  // namespace cldnn
