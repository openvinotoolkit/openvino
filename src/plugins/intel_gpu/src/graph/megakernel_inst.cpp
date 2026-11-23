// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "megakernel_inst.h"

#include "intel_gpu/op/megakernel.hpp"
#include "json_object.h"
#include "primitive_type_base.h"

#include <sstream>

namespace cldnn {

GPU_DEFINE_PRIMITIVE_TYPE_ID(megakernel)

template <typename ShapeType>
std::vector<layout> megakernel_inst::calc_output_layouts(megakernel_node const& /*node*/,
                                                                 const kernel_impl_params& impl_param) {
    auto desc = impl_param.typed_desc<megakernel>();

    std::vector<ov::PartialShape> input_shapes;
    for (const auto& l : impl_param.input_layouts)
        input_shapes.push_back(l.get<ov::PartialShape>());

    std::vector<layout> outs;
    for (size_t i = 0; i < desc->attrs.outputs.size(); ++i) {
        const auto& out = desc->attrs.outputs[i];
        const auto type = desc->output_data_types[i].value_or(out.type);
        outs.emplace_back(ov::intel_gpu::op::megakernel_output_shape(out, input_shapes), type,
                          format::get_default_format(out.shape.size()));
    }
    return outs;
}

layout megakernel_inst::calc_output_layout(megakernel_node const& node,
                                                   kernel_impl_params const& impl_param) {
    return calc_output_layouts<ov::PartialShape>(node, impl_param)[0];
}

template std::vector<layout> megakernel_inst::calc_output_layouts<ov::PartialShape>(
    megakernel_node const& node,
    const kernel_impl_params& impl_param);

// ---------------------------------------------------------------------------
// to_string / constructor
// ---------------------------------------------------------------------------
std::string megakernel_inst::to_string(megakernel_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    std::stringstream ss;
    json_composite info;
    info.add("num_outputs", desc->attrs.outputs.size());
    info.add("params_bytes", desc->attrs.params.size());
    node_info->add("megakernel_info", info);
    node_info->dump(ss);
    return ss.str();
}

megakernel_inst::typed_primitive_inst(network& network, megakernel_node const& node)
    : parent(network, node) {}

}  // namespace cldnn
