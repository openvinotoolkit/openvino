// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>

#include "fully_connected_gguf_inst.h"
#include "json_object.h"
#include "primitive_type_base.h"
#include "to_string_utils.h"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(fully_connected_gguf)

layout fully_connected_gguf_inst::calc_output_layout(const fully_connected_gguf_node& node, const kernel_impl_params& impl_param) {
    return calc_output_layouts<ov::PartialShape>(node, impl_param)[0];
}

template <typename ShapeType>
std::vector<layout> fully_connected_gguf_inst::calc_output_layouts(const fully_connected_gguf_node& /*node*/, const kernel_impl_params& impl_param) {
    const auto desc = impl_param.typed_desc<fully_connected_gguf>();
    const auto& input_layout = impl_param.get_input_layout(0);
    const auto& weights_shape = impl_param.get_input_layout(1).get_partial_shape();

    // [..., M, K] x [N, bytes] -> [..., M, N]
    auto output_shape = input_layout.get_partial_shape();
    if (output_shape.rank().is_static() && output_shape.size() > 0) {
        output_shape[output_shape.size() - 1] = weights_shape.rank().is_static() ? weights_shape[0] : ov::Dimension::dynamic();
    }
    const auto output_type = desc->output_data_types[0].value_or(input_layout.data_type);
    return {layout{output_shape, output_type, format::get_default_format(output_shape.size())}};
}

template std::vector<layout> fully_connected_gguf_inst::calc_output_layouts<ov::PartialShape>(const fully_connected_gguf_node& node,
                                                                                              const kernel_impl_params& impl_param);

std::string fully_connected_gguf_inst::to_string(const fully_connected_gguf_node& node) {
    auto node_info = node.desc_to_json();
    auto desc = node.get_primitive();

    std::stringstream primitive_description;
    json_composite info;
    info.add("format", ov::as_string(desc->format));
    node_info->add("fully_connected_gguf_info", info);
    node_info->dump(primitive_description);
    return primitive_description.str();
}

fully_connected_gguf_inst::typed_primitive_inst(network& network, const fully_connected_gguf_node& node) : parent(network, node) {}
}  // namespace cldnn
