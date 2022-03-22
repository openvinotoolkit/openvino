// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "binary_convolution_inst.h"
#include "convolution_inst.h"
#include "reorder_inst.h"
#include "primitive_type_base.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include <string>

namespace cldnn {
primitive_type_id binary_convolution::type_id() {
    static primitive_type_base<binary_convolution> instance;
    return &instance;
}

layout binary_convolution_inst::calc_output_layout(binary_convolution_node const& node) {
    auto desc = node.get_primitive();

    auto output_type = *node.get_primitive()->output_data_type;
    auto output_size = desc->output_size;
    auto layout = cldnn::layout{output_type, format::bfyx, output_size};
    if (node.has_fused_primitives()) {
        layout = node.get_fused_output_layout();
    }

    auto users = node.get_users();
    if (users.size() == 1 && users.front()->is_type<convolution>()) {
        auto conv_split = users.front()->as<convolution>().get_split();
        auto conv_groups = (int32_t)users.front()->as<convolution>().get_groups();

        bool next_is_dw = ((conv_split > 1 && conv_split == output_size.feature[0]) ||
                           (conv_groups > 1 && conv_groups == output_size.feature[0]));

        if ((layout.data_type == data_types::f16 || layout.data_type == data_types::f32) && next_is_dw) {
            layout.format = cldnn::format::b_fs_yx_fsv16;
        }
    }

    return layout;
}

std::string binary_convolution_inst::to_string(binary_convolution_node const& node) {
    auto desc = node.get_primitive();
    auto strd = desc->stride;
    auto split = node.get_split();
    auto dilation = desc->dilation;
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;

    json_composite conv_info;
    conv_info.add("stride", cldnn::to_string(strd));
    conv_info.add("pad", cldnn::to_string(desc->pad));
    conv_info.add("split", split);
    conv_info.add("dilation", cldnn::to_string(dilation));
    conv_info.add("out size", desc->output_size.to_string());

    node_info->add("binary convolution info", conv_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

binary_convolution_inst::typed_primitive_inst(network& network, binary_convolution_node const& node)
    : parent(network, node) {
    auto stride = argument.stride;
    auto pad = argument.pad;

    auto input_layout = node.input().get_output_layout();
    auto output_layout = node.get_output_layout();
    auto output_size = output_layout.size;

    CLDNN_ERROR_NOT_EQUAL(node.id(),
                          "Input number of dimensions",
                          input_layout.size.raw.size(),
                          "output number of dimensions",
                          output_layout.size.raw.size(),
                          "Input/output dims mismatch");
    CLDNN_ERROR_NOT_EQUAL(node.id(),
                          "Stride number of dimensions",
                          stride.size(),
                          "output number of dimensions",
                          output_layout.get_spatial_rank(),
                          "stride/output dims mismatch");
    CLDNN_ERROR_NOT_EQUAL(node.id(),
                          "pad number of dimensions",
                          pad.size(),
                          "input number of dimensions",
                          input_layout.get_spatial_rank(),
                          "Input offset/ input size mismatch");

    auto split = node.get_split();
    for (decltype(split) j = 0; j < split; j++) {
        auto filter_inst = node.weights(j).get_output_layout();  // convolution filter

        auto pad = argument.pad;

        CLDNN_ERROR_NOT_EQUAL(node.id(),
                              "Weights number of dimensions",
                              filter_inst.size.raw.size(),
                              "output number of dimensions",
                              output_layout.size.raw.size(),
                              "Weights/output dims mismatch");
        CLDNN_ERROR_NOT_EQUAL(node.id(),
                              "Convolution padding mode",
                              node.get_output_layout().data_padding.filling_value(),
                              "padding value",
                              0.0f,
                              "Unknown padding mode.");
        CLDNN_ERROR_NOT_EQUAL(node.id(),
                              "Output feature size",
                              output_size.feature.size(),
                              "expected feature size",
                              1,
                              "Only one-dimensional features are supported");
        CLDNN_ERROR_NOT_EQUAL(node.id(),
                              "Output batch size",
                              output_size.batch.size(),
                              "expected output size",
                              1,
                              "Only one-dimensional batch size are supported");
        CLDNN_ERROR_LESS_THAN(node.id(),
                              "Weights feature maps number",
                              input_layout.size.feature[0],
                              "input feature maps number",
                              filter_inst.size.feature[0],
                              "Weights/ifm mismatch");
    }
}
}  // namespace cldnn
