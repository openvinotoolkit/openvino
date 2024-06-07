// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "extract_image_patches_inst.h"

#include "openvino/op/extractimagepatches.hpp"
#include "extract_image_patches_shape_inference.hpp"
#include "primitive_type_base.h"
#include "json_object.h"
#include <string>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(extract_image_patches)

template<typename ShapeType>
std::vector<layout> extract_image_patches_inst::calc_output_layouts(extract_image_patches_node const& /*node*/, const kernel_impl_params& impl_param) {
    const auto& input_layout = impl_param.get_input_layout();
    auto primitive = impl_param.typed_desc<extract_image_patches>();
    ov::op::v3::ExtractImagePatches op;
    op.set_sizes(primitive->sizes);
    op.set_rates(primitive->rates);
    op.set_strides(primitive->strides);
    op.set_auto_pad(primitive->auto_pad);
    auto out_shapes = ov::op::v3::shape_infer(&op, std::vector<ShapeType>{input_layout.get<ShapeType>()});
    return { layout{ out_shapes[0], input_layout.data_type, input_layout.format} };
}

template std::vector<layout>
extract_image_patches_inst::calc_output_layouts<ov::PartialShape>(extract_image_patches_node const& node, const kernel_impl_params& impl_param);

layout extract_image_patches_inst::calc_output_layout(extract_image_patches_node const& node, kernel_impl_params const& impl_param) {
    auto desc = impl_param.typed_desc<extract_image_patches>();

    auto input_layout = impl_param.get_input_layout();
    auto input_format = input_layout.format;

    auto output_shape = desc->output_shape;
    return layout(input_layout.data_type, input_format, output_shape);
}

std::string extract_image_patches_inst::to_string(extract_image_patches_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto& input = node.input();

    std::stringstream primitive_description;

    std::stringstream sizes, strides, rates;
    sizes << desc->sizes[0] << "," << desc->sizes[1];
    strides << desc->strides[0] << "," << desc->strides[1];
    rates << desc->rates[0] << "," << desc->rates[1];

    json_composite extract_image_patches_info;
    extract_image_patches_info.add("input id", input.id());
    extract_image_patches_info.add("sizes", sizes.str());
    extract_image_patches_info.add("strides", strides.str());
    extract_image_patches_info.add("rates", rates.str());
    extract_image_patches_info.add("auto_pad", desc->auto_pad);

    node_info->add("extract_image_patches info", extract_image_patches_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

extract_image_patches_inst::typed_primitive_inst(network& network, extract_image_patches_node const& node) : parent(network, node) {}

}  // namespace cldnn
