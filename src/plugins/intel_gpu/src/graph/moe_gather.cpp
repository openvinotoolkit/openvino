// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "moe_gather_inst.h"
#include "primitive_type_base.h"
#include "json_object.h"
#include "to_string_utils.h"
#include <string>
#include <vector>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(moe_gather)

layout moe_gather_inst::calc_output_layout(moe_gather_node const& node, kernel_impl_params const& impl_param) {
    auto output_layouts = calc_output_layouts<ov::PartialShape>(node, impl_param);
    return output_layouts[0];
}

template <typename ShapeType>
std::vector<layout> moe_gather_inst::calc_output_layouts(const moe_gather_node& /*node*/, const kernel_impl_params& impl_param) {
    const auto& desc = impl_param.typed_desc<moe_gather>();
    const auto num_experts_per_token = desc->num_experts_per_token;
    const auto& in_layout = impl_param.input_layouts[0];
    const auto& input_shape = in_layout.get<ShapeType>();
    OPENVINO_ASSERT(input_shape.size() == 2,
                    impl_param.desc->id, " expects rank-2 [N_tokens, hidden] input, got rank ", input_shape.size());
    OPENVINO_ASSERT(input_shape[1].is_static(),
                    impl_param.desc->id, " hidden size dimension must be static");

    // [N, H] -> [N*K, H].
    auto out_shape = input_shape;
    if (input_shape[0].is_static()) {
        out_shape[0] = ov::Dimension(input_shape[0].get_length() * num_experts_per_token);
    }
    return {layout{out_shape, in_layout.data_type, in_layout.format}};
}

template std::vector<layout> moe_gather_inst::calc_output_layouts<ov::PartialShape>(moe_gather_node const& node, const kernel_impl_params& impl_param);

std::string moe_gather_inst::to_string(moe_gather_node const& node) {
    auto node_info = node.desc_to_json();
    auto desc = node.get_primitive();

    std::stringstream primitive_description;

    json_composite moe_gather_info;
    if (desc->output_data_types[0].has_value())
        moe_gather_info.add("out dt: ", dt_to_str(*desc->output_data_types[0]));
    node_info->add("moe_gather info", moe_gather_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

moe_gather_inst::typed_primitive_inst(network& network, moe_gather_node const& node) : parent(network, node) { }
}  // namespace cldnn
