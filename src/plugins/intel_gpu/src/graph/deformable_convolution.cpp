// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "deformable_convolution_inst.h"
#include "primitive_type_base.h"
#include "json_object.h"
#include <string>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(deformable_conv)

layout deformable_conv_inst::calc_output_layout(deformable_conv_node const& node, kernel_impl_params const& impl_param) {
    auto desc = impl_param.typed_desc<deformable_conv>();

    auto input_layout = impl_param.get_input_layout();

    auto input_type = input_layout.data_type;
    auto output_type = desc->output_data_types[0].value_or(input_type);

    tensor output_size(input_layout.batch(),
                       desc->output_size.feature[0],
                       desc->output_size.spatial[0],
                       desc->output_size.spatial[1],
                       desc->output_size.spatial[2]);

    return {output_type, input_layout.format, output_size};
}

std::string deformable_conv_inst::to_string(deformable_conv_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;

    json_composite conv_info;
    conv_info.add("groups", desc->groups);

    json_composite ud_out_size_info;
    ud_out_size_info.add("size", desc->output_size.to_string());
    conv_info.add("with user defined output size", ud_out_size_info);

    node_info->add("deformable_convolution info", conv_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

deformable_conv_inst::typed_primitive_inst(network& network, deformable_conv_node const& node) : parent(network, node) {
}

GPU_DEFINE_PRIMITIVE_TYPE_ID(deformable_interp)

layout deformable_interp_inst::calc_output_layout(deformable_interp_node const& node, kernel_impl_params const& impl_param) {
    auto desc = node.get_primitive();

    auto input_layout = node.input().get_output_layout();

    auto kernel_size = desc->kernel_size;
    auto input_type = input_layout.data_type;
    auto output_type = node.get_primitive()->output_data_types[0].value_or(input_type);

    tensor output_size(input_layout.batch(),
                       input_layout.feature()*kernel_size.spatial[0]*kernel_size.spatial[1],
                       desc->output_size.spatial[0],
                       desc->output_size.spatial[1],
                       desc->output_size.spatial[2]);

    return {output_type, input_layout.format, output_size};
}

std::string deformable_interp_inst::to_string(deformable_interp_node const& node) {
    auto desc = node.get_primitive();
    auto strd = desc->stride;
    auto dilation = desc->dilation;
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;

    json_composite interp_info;
    interp_info.add("stride", cldnn::to_string(strd));
    interp_info.add("pad", cldnn::to_string(desc->pad));
    interp_info.add("dilation", cldnn::to_string(dilation));
    interp_info.add("deformable_groups", desc->deformable_groups);
    interp_info.add("groups", desc->groups);
    interp_info.add("bilinear_interpolation_pad", desc->bilinear_interpolation_pad);

    json_composite ud_out_size_info;
    ud_out_size_info.add("size", desc->output_size.to_string());
    interp_info.add("with user defined output size", ud_out_size_info);

    node_info->add("deformable_interpolation info", interp_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

deformable_interp_inst::typed_primitive_inst(network& network, deformable_interp_node const& node) : parent(network, node) {
}

}  // namespace cldnn
