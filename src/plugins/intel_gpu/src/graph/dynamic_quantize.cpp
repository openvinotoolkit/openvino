// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/dynamic_quantize.hpp"
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
                                                                 const dynamic_quantize::QuantizationConfig& config,
                                                                 const std::vector<uint64_t>& scales_zp_output_order,
                                                                 const bool combine_scales_and_zp) {
    ov::intel_gpu::op::DynamicQuantize op;
    auto output_format = act_layout.format;

    std::vector<ShapeType> input_shapes = {
        act_layout.get<ShapeType>(),
    };

    auto output_shapes = ov::intel_gpu::op::DynamicQuantize::shape_infer(&op, input_shapes, config, scales_zp_output_order, combine_scales_and_zp);

    std::vector<layout> output_layouts = { layout(output_shapes[0], config.quantization_dt, output_format),
                                           layout(output_shapes[1], config.scale_dt, output_format) };

    if (config.is_asymmetric_quantization() && !combine_scales_and_zp) {
        output_layouts.emplace_back(layout(output_shapes[2], config.zp_dt, output_format));
    }

    return output_layouts;
}

template std::vector<layout> dynamic_quantize_inst::__calc_output_layouts<ov::PartialShape>(const layout &act_layout,
                                                                                            const dynamic_quantize::QuantizationConfig& config,
                                                                                            const std::vector<uint64_t>& scales_zp_output_order,
                                                                                            const bool combine_scales_and_zp);

template<typename ShapeType>
std::vector<layout> dynamic_quantize_inst::calc_output_layouts(dynamic_quantize_node const& /*node*/, const kernel_impl_params& impl_param) {
    auto desc = impl_param.typed_desc<dynamic_quantize>();
    const auto& input_layout = impl_param.get_input_layout();

    return __calc_output_layouts<ov::PartialShape>(input_layout, desc->quantization_config, desc->scales_zp_output_order, desc->combine_scales_and_zp);
}

template std::vector<layout> dynamic_quantize_inst::calc_output_layouts<ov::PartialShape>(dynamic_quantize_node const& node,
                                                                                const kernel_impl_params& impl_param);

std::string dynamic_quantize_inst::to_string(dynamic_quantize_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;

    json_composite dynamic_quantize_info;
    dynamic_quantize_info.add("combine_scales_and_zp", desc->combine_scales_and_zp);
    dynamic_quantize_info.add("scales_zp_output_order", desc->scales_zp_output_order);
    dynamic_quantize_info.add("quantization_dt", desc->quantization_config.quantization_dt);
    dynamic_quantize_info.add("scale_dt", desc->quantization_config.scale_dt);
    dynamic_quantize_info.add("zp_dt", desc->quantization_config.zp_dt);
    dynamic_quantize_info.add("is_asymmetric_quantization", desc->quantization_config.is_asymmetric_quantization());
    node_info->add("dynamic_quantize info", dynamic_quantize_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

dynamic_quantize_inst::typed_primitive_inst(network& network, dynamic_quantize_node const& node) : parent(network, node) {}

}  // namespace cldnn
