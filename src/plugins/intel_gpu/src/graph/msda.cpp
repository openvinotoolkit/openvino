// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "json_object.h"
#include "msda_inst.h"
#include "primitive_type_base.h"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(msda);

namespace {

template <typename ShapeType>
std::vector<layout> msda_inst::calc_output_layouts(const msda_node& /*node*/, const kernel_impl_params& impl_param) {
    auto desc = impl_param.typed_desc<msda>();
    auto feat_value_input_layout = impl_param.get_input_layout(0);
    auto attn_weights_input_layout = impl_param.get_input_layout(4);

    auto output_type = feat_value_input_layout.data_type;
    if (impl_param.has_fused_primitives()) {
        output_type = impl_param.get_output_element_type();
    }

    const auto feat_value_ps = feat_value_input_layout.get_partial_shape();
    const auto attn_weight_ps = attn_weights_input_layout.get_partial_shape();
    auto output_shape = ov::PartialShape({feat_value_ps[0], attn_weight_ps[1], feat_value_ps[2] * feat_value_ps[3]});

    format output_format = format::adjust_to_rank(feat_value_input_layout.format, output_shape.size());
    return {layout{output_shape, output_type, output_format}};
}

std::string msda_inst::to_string(const msda_node& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;
    node_info->dump(primitive_description);

    return primitive_description.str();
}

msda_inst::typed_primitive_inst(network& network, const msda_node& node) : parent(network, node) {}

}  // namespace cldnn