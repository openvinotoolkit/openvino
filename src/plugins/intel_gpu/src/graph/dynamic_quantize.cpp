// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_ops/dynamic_quantize.hpp"
#include "dynamic_quantize_inst.h"

#include "primitive_type_base.h"
#include "json_object.h"
#include <string>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(dynamic_quantize);

layout dynamic_quantize_inst::calc_output_layout(dynamic_quantize_node const& node, kernel_impl_params const& impl_param) {
    auto desc = impl_param.typed_desc<dynamic_quantize>();
    const auto& input_layout = impl_param.get_input_layout();
    auto output_type = data_types::i8;
    auto output_format = input_layout.format;

    return layout(output_type, output_format, input_layout.get_tensor());
}

template<typename ShapeType>
std::vector<layout> dynamic_quantize_inst::__calc_output_layouts(const layout &act_layout,
                                                                 const dynamic_quantize::Attributes& attrs) {
    ov::op::internal::DynamicQuantize op;
    op.set_attrs(attrs);

    auto output_format = act_layout.format;

    std::vector<ShapeType> input_shapes = {
        act_layout.get<ShapeType>(),
    };

    auto output_shapes = ov::op::internal::DynamicQuantize::shape_infer(&op, input_shapes);

    std::vector<layout> output_layouts = { layout(output_shapes[0], attrs.quantization_dt, output_format),
                                           layout(output_shapes[1], attrs.scale_dt, output_format) };

    if (attrs.quantization_type == ov::op::internal::DynamicQuantize::QuantizationType::Asymmetric &&
        attrs.output_storage_type == ov::op::internal::DynamicQuantize::OutputStorageType::Planar) {
        output_layouts.emplace_back(layout(output_shapes[2], attrs.zp_dt, output_format));
    }

    return output_layouts;
}

template std::vector<layout> dynamic_quantize_inst::__calc_output_layouts<ov::PartialShape>(const layout &act_layout,
                                                                                            const dynamic_quantize::Attributes& config);

template<typename ShapeType>
std::vector<layout> dynamic_quantize_inst::calc_output_layouts(dynamic_quantize_node const& /*node*/, const kernel_impl_params& impl_param) {
    auto desc = impl_param.typed_desc<dynamic_quantize>();
    const auto& input_layout = impl_param.get_input_layout();

    return __calc_output_layouts<ov::PartialShape>(input_layout, desc->attrs);
}

template std::vector<layout> dynamic_quantize_inst::calc_output_layouts<ov::PartialShape>(dynamic_quantize_node const& node,
                                                                                const kernel_impl_params& impl_param);

std::string dynamic_quantize_inst::to_string(dynamic_quantize_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;

    json_composite dynamic_quantize_info;
    dynamic_quantize_info.add("output_storage_type", static_cast<int>(desc->attrs.output_storage_type));
    dynamic_quantize_info.add("scales_zp_output_order", desc->attrs.scales_zp_output_order);
    dynamic_quantize_info.add("group_sizes", desc->attrs.group_sizes);
    dynamic_quantize_info.add("quantization_dt", desc->attrs.quantization_dt);
    dynamic_quantize_info.add("scale_dt", desc->attrs.scale_dt);
    dynamic_quantize_info.add("zp_dt", desc->attrs.zp_dt);
    dynamic_quantize_info.add("quantization_type", static_cast<int>(desc->attrs.quantization_type));
    node_info->add("dynamic_quantize info", dynamic_quantize_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

dynamic_quantize_inst::typed_primitive_inst(network& network, dynamic_quantize_node const& node) : parent(network, node) {}

}  // namespace cldnn
