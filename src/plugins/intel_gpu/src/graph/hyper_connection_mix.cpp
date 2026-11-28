// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>

#include "hyper_connection_mix_inst.h"
#include "json_object.h"
#include "primitive_type_base.h"
#include "to_string_utils.h"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(hyper_connection_mix)

layout hyper_connection_mix_inst::calc_output_layout(const hyper_connection_mix_node& node, const kernel_impl_params& impl_param) {
    return calc_output_layouts<ov::PartialShape>(node, impl_param)[0];
}

template <typename ShapeType>
std::vector<layout> hyper_connection_mix_inst::calc_output_layouts(const hyper_connection_mix_node& /*node*/, const kernel_impl_params& impl_param) {
    const auto& desc = impl_param.typed_desc<hyper_connection_mix>();
    const auto in = impl_param.get_input_layout(0);
    ov::PartialShape vec = in.get_partial_shape();
    ov::PartialShape comb = vec;
    const auto n = desc->n_streams;
    if (vec.rank().is_static()) {
        const auto r = vec.rank().get_length();
        vec[r - 1] = n;
        comb[r - 1] = n;
        comb.push_back(n);
    }
    const auto fmt_vec = format::get_default_format(vec.size());
    const auto fmt_comb = format::get_default_format(comb.size());
    return {layout{vec, in.data_type, fmt_vec}, layout{vec, in.data_type, fmt_vec}, layout{comb, in.data_type, fmt_comb}};
}

template std::vector<layout> hyper_connection_mix_inst::calc_output_layouts<ov::PartialShape>(const hyper_connection_mix_node& node,
                                                                                              const kernel_impl_params& impl_param);

std::string hyper_connection_mix_inst::to_string(const hyper_connection_mix_node& node) {
    auto node_info = node.desc_to_json();
    auto desc = node.get_primitive();
    std::stringstream primitive_description;
    json_composite info;
    info.add("mixes", node.input(0).id());
    info.add("n_streams", desc->n_streams);
    info.add("sinkhorn_iterations", desc->sinkhorn_iterations);
    info.add("eps", desc->eps);
    node_info->add("hyper_connection_mix_info", info);
    node_info->dump(primitive_description);
    return primitive_description.str();
}

hyper_connection_mix_inst::typed_primitive_inst(network& network, const hyper_connection_mix_node& node) : parent(network, node) {}
}  // namespace cldnn
