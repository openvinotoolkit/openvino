// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mvn_inst.h"
#include "primitive_type_base.h"
#include "json_object.h"
#include <algorithm>
#include <string>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(mvn)

bool mvn::is_aligned_layout_supported(const layout& input_layout) const {
    const auto& input_pshape = input_layout.get_partial_shape();
    if (!requires_alignment(input_pshape))
        return true;

    const auto& fmt = input_layout.format;
    if (format::is_default_format(fmt))
        return true;

    //defer to dyn_formats
    if (input_pshape.is_dynamic())
        return true;

    // Mirror the single feature-blocked case handled by mvn_impl::static_canonicalize_shapes.
    const auto& block_sizes = format::block_sizes(fmt);
    auto axes = reduction_axes;
    const auto rank = static_cast<int64_t>(input_pshape.size());
    std::for_each(axes.begin(), axes.end(), [rank](int64_t& v) {
        v = (v < 0) ? v + rank : v;
    });
    return block_sizes.size() == 1 && block_sizes[0].first == 1 &&
           (input_pshape[block_sizes[0].first].get_length() % block_sizes[0].second == 0) &&
           (std::count(axes.begin(), axes.end(), static_cast<int64_t>(block_sizes[0].first)) == 0);
}

layout mvn_inst::calc_output_layout(mvn_node const& node, kernel_impl_params const& impl_param) {
    auto input_node_layout = impl_param.get_non_padded_input_layout();
    auto output_type = impl_param.desc->output_data_types[0].value_or(input_node_layout.data_type);

    if (impl_param.has_fused_primitives()) {
        output_type = impl_param.get_output_element_type();
    } else if (input_node_layout.data_type == data_types::u8 || input_node_layout.data_type == data_types::i8) {
        output_type = data_types::f32;
    }

    return layout(output_type, input_node_layout.format, input_node_layout.get_tensor());
}

std::string mvn_inst::to_string(mvn_node const& node) {
    auto node_info = node.desc_to_json();
    auto desc = node.get_primitive();
    auto epsilon = desc->epsilon;
    auto axes = desc->reduction_axes;
    auto normalize_variance = desc->normalize_variance ? "true" : "false";
    auto eps_inside_sqrt = desc->eps_inside_sqrt ? "true" : "false";
    auto& input = node.input();

    std::stringstream primitive_description;

    json_composite mvn_info;
    mvn_info.add("input id", input.id());
    mvn_info.add("epsilon", epsilon);
    mvn_info.add("reduction axes", std::move(axes));
    mvn_info.add("normalize_variance region", normalize_variance);
    mvn_info.add("eps_inside_sqrt region", eps_inside_sqrt);

    node_info->add("mvn info", mvn_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

mvn_inst::typed_primitive_inst(network& network, mvn_node const& node) : parent(network, node) {}
}  // namespace cldnn
