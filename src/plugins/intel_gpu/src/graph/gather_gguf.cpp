// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gather_gguf_inst.h"
#include "primitive_type_base.h"
#include "json_object.h"
#include "to_string_utils.h"
#include <string>
#include <vector>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(gather_gguf)

layout gather_gguf_inst::calc_output_layout(gather_gguf_node const& node, kernel_impl_params const& impl_param) {
    auto output_layouts = calc_output_layouts<ov::PartialShape>(node, impl_param);
    return output_layouts[0];
}

template <typename ShapeType>
std::vector<layout> gather_gguf_inst::calc_output_layouts(const gather_gguf_node& /*node*/,
                                                          const kernel_impl_params& impl_param) {
    const auto& desc = impl_param.typed_desc<gather_gguf>();
    const auto& indices_layout = impl_param.input_layouts[1];
    const auto& indices_shape = indices_layout.get<ShapeType>();
    const auto hidden = ov::Dimension(desc->hidden_size);

    // Output = gather(axis=0) of [vocab, hidden] with `indices` => [*indices_dims, hidden], always f16.
    ShapeType out_shape = indices_shape;
    out_shape.push_back(hidden);

    const auto out_format = format::get_default_format(out_shape.size());
    return {layout{out_shape, data_types::f16, out_format}};
}

template std::vector<layout> gather_gguf_inst::calc_output_layouts<ov::PartialShape>(gather_gguf_node const& node,
                                                                                     const kernel_impl_params& impl_param);

std::string gather_gguf_inst::to_string(gather_gguf_node const& node) {
    auto node_info = node.desc_to_json();
    auto desc = node.get_primitive();

    std::stringstream primitive_description;

    json_composite gather_gguf_info;
    gather_gguf_info.add("weight_type", desc->weight_type.get_type_name());
    gather_gguf_info.add("vocab_size", desc->vocab_size);
    gather_gguf_info.add("hidden_size", desc->hidden_size);
    node_info->add("gather_gguf info", gather_gguf_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

gather_gguf_inst::typed_primitive_inst(network& network, gather_gguf_node const& node) : parent(network, node) {}
}  // namespace cldnn
