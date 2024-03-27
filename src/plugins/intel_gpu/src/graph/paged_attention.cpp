// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "paged_attention_inst.h"
#include "primitive_type_base.h"
#include "json_object.h"

#include <string>
#include <sstream>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(paged_attention)


layout paged_attention_inst::calc_output_layout(const paged_attention_node& node, kernel_impl_params const& impl_param) {
    return impl_param.input_layouts[0];
}

template<typename ShapeType>
std::vector<layout> paged_attention_inst::calc_output_layouts(paged_attention_node const& /*node*/, kernel_impl_params const& impl_param) {
    auto desc = impl_param.typed_desc<paged_attention>();

    auto out_layout = impl_param.get_input_layout(0);

    return {out_layout};
}

template std::vector<layout> paged_attention_inst::calc_output_layouts<ov::PartialShape>(paged_attention_node const& node, const kernel_impl_params& impl_param);

std::string paged_attention_inst::to_string(const paged_attention_node& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;

    json_composite custom_gpu_prim_info;
    node_info->add("paged attention primitive info", custom_gpu_prim_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

void paged_attention_inst::update_shape_info_tensor(const kernel_impl_params& params) {
    parent::update_shape_info_tensor(params);
}

paged_attention_inst::typed_primitive_inst(network& network, const paged_attention_node& node)
    : parent(network, node) {}
}  // namespace cldnn
