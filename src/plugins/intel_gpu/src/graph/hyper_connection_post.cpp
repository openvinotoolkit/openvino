// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>

#include "hyper_connection_post_inst.h"
#include "json_object.h"
#include "primitive_type_base.h"
#include "to_string_utils.h"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(hyper_connection_post)

layout hyper_connection_post_inst::calc_output_layout(const hyper_connection_post_node& node, const kernel_impl_params& impl_param) {
    return calc_output_layouts<ov::PartialShape>(node, impl_param)[0];
}

template <typename ShapeType>
std::vector<layout> hyper_connection_post_inst::calc_output_layouts(const hyper_connection_post_node& /*node*/, const kernel_impl_params& impl_param) {
    const auto res = impl_param.get_input_layout(1);
    return {layout{res.get_partial_shape(), res.data_type, format::get_default_format(res.get_partial_shape().size())}};
}

template std::vector<layout> hyper_connection_post_inst::calc_output_layouts<ov::PartialShape>(const hyper_connection_post_node& node,
                                                                                               const kernel_impl_params& impl_param);

std::string hyper_connection_post_inst::to_string(const hyper_connection_post_node& node) {
    auto node_info = node.desc_to_json();
    std::stringstream primitive_description;
    json_composite info;
    info.add("y", node.input(0).id());
    info.add("residual", node.input(1).id());
    node_info->add("hyper_connection_post_info", info);
    node_info->dump(primitive_description);
    return primitive_description.str();
}

hyper_connection_post_inst::typed_primitive_inst(network& network, const hyper_connection_post_node& node) : parent(network, node) {}
}  // namespace cldnn
