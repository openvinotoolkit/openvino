// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>

#include "hyper_connection_pre_inst.h"
#include "json_object.h"
#include "primitive_type_base.h"
#include "to_string_utils.h"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(hyper_connection_pre)

layout hyper_connection_pre_inst::calc_output_layout(const hyper_connection_pre_node& node, const kernel_impl_params& impl_param) {
    return calc_output_layouts<ov::PartialShape>(node, impl_param)[0];
}

template <typename ShapeType>
std::vector<layout> hyper_connection_pre_inst::calc_output_layouts(const hyper_connection_pre_node& /*node*/,
                                                                   const kernel_impl_params& impl_param) {
    const auto& desc = impl_param.typed_desc<hyper_connection_pre>();
    const auto in = impl_param.get_input_layout(0);  // streams [..., n, e]
    const auto& s = in.get_partial_shape();
    const auto n = desc->n_streams;
    ov::PartialShape y, vec, comb;
    if (s.rank().is_static()) {
        const auto r = s.rank().get_length();
        for (int64_t i = 0; i + 2 < r; i++) {
            y.push_back(s[i]);
            vec.push_back(s[i]);
            comb.push_back(s[i]);
        }
        y.push_back(s[r - 1]);
        vec.push_back(n);
        comb.push_back(n);
        comb.push_back(n);
    } else {
        y = vec = comb = ov::PartialShape::dynamic();
    }
    return {layout{y, in.data_type, format::get_default_format(y.size())},
            layout{vec, in.data_type, format::get_default_format(vec.size())},
            layout{comb, in.data_type, format::get_default_format(comb.size())}};
}

template std::vector<layout> hyper_connection_pre_inst::calc_output_layouts<ov::PartialShape>(const hyper_connection_pre_node& node,
                                                                                              const kernel_impl_params& impl_param);

std::string hyper_connection_pre_inst::to_string(const hyper_connection_pre_node& node) {
    auto node_info = node.desc_to_json();
    auto desc = node.get_primitive();
    std::stringstream primitive_description;
    json_composite info;
    info.add("streams", node.input(0).id());
    info.add("n_streams", desc->n_streams);
    info.add("sinkhorn_iterations", desc->sinkhorn_iterations);
    info.add("eps", desc->eps);
    info.add("rms_eps", desc->rms_eps);
    node_info->add("hyper_connection_pre_info", info);
    node_info->dump(primitive_description);
    return primitive_description.str();
}

hyper_connection_pre_inst::typed_primitive_inst(network& network, const hyper_connection_pre_node& node) : parent(network, node) {}
}  // namespace cldnn
