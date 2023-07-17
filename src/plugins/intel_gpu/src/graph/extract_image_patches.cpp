// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "extract_image_patches_inst.h"

#include "primitive_type_base.h"
#include "json_object.h"
#include <string>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(extract_image_patches)

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
