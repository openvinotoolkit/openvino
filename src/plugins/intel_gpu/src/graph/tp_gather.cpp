// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef ENABLE_TP_GPU

#include "tp_gather_inst.h"
#include "primitive_type_base.h"
#include "json_object.h"

#include <string>

namespace cldnn {

GPU_DEFINE_PRIMITIVE_TYPE_ID(tp_gather)

namespace {

/// Output layout of a gather: the gathered axis grows by world_size on the
/// root, and stays as it is everywhere else -- only the root is written.
layout gathered_layout(const layout& input_layout, const tp_gather& desc) {
    if (desc.rank != 0) {
        return input_layout;
    }
    auto pshape = input_layout.get_partial_shape();
    if (pshape.rank().is_dynamic()) {
        return input_layout;
    }
    const auto rank_len = pshape.rank().get_length();
    const auto axis = desc.axis >= 0 ? desc.axis : desc.axis + rank_len;
    OPENVINO_ASSERT(axis >= 0 && axis < rank_len,
                    "[GPU] tp_gather axis ", desc.axis, " is out of range for a rank-", rank_len, " input");
    if (pshape[axis].is_static()) {
        pshape[axis] = pshape[axis].get_length() * static_cast<int64_t>(desc.world_size);
    } else {
        pshape[axis] = ov::Dimension::dynamic();
    }
    return layout(pshape, input_layout.data_type, input_layout.format);
}

}  // namespace

layout tp_gather_inst::calc_output_layout(const tp_gather_node& node, const kernel_impl_params& impl_param) {
    return gathered_layout(impl_param.get_input_layout(), *impl_param.typed_desc<tp_gather>());
}

template <typename ShapeType>
std::vector<layout> tp_gather_inst::calc_output_layouts(const tp_gather_node& node, const kernel_impl_params& impl_param) {
    return {gathered_layout(impl_param.get_input_layout(), *impl_param.typed_desc<tp_gather>())};
}

template std::vector<layout>
tp_gather_inst::calc_output_layouts<ov::PartialShape>(const tp_gather_node& node, const kernel_impl_params& impl_param);

std::string tp_gather_inst::to_string(const tp_gather_node& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();

    json_composite tp_info;
    tp_info.add("collective_id", desc->collective_id);
    tp_info.add("rank", desc->rank);
    tp_info.add("world_size", desc->world_size);
    tp_info.add("axis", desc->axis);

    node_info->add("tp_gather params", tp_info);
    std::stringstream primitive_description;
    node_info->dump(primitive_description);
    return primitive_description.str();
}

tp_gather_inst::typed_primitive_inst(network& network, const tp_gather_node& node)
    : parent(network, node) {}

}  // namespace cldnn

#endif  // ENABLE_TP_GPU
