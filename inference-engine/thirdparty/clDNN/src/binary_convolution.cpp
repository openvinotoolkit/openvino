/*
// Copyright (c) 2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "binary_convolution_inst.h"
#include "convolution_inst.h"
#include "reorder_inst.h"
#include "primitive_type_base.h"
#include "sliding_window_utils.h"
#include "error_handler.h"
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
            layout.format = cldnn::format::bfyx_f16;
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
    conv_info.add("stride", strd.to_string());
    conv_info.add("input offset", desc->input_offset.to_string());
    conv_info.add("split", split);
    conv_info.add("dilation", dilation.to_string());
    conv_info.add("out size", desc->output_size.to_string());

    node_info->add("binary convolution info", conv_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

binary_convolution_inst::typed_primitive_inst(network_impl& network, binary_convolution_node const& node)
    : parent(network, node) {
    auto stride = argument.stride;

    auto input_inst = node.input().get_output_layout();
    auto output_inst = node.get_output_layout();
    auto output_size = output_inst.size;

    CLDNN_ERROR_NOT_EQUAL(node.id(),
                          "Input number of dimensions",
                          input_inst.size.raw.size(),
                          "output number of dimensions",
                          output_inst.size.raw.size(),
                          "Input/output dims mismatch");
    CLDNN_ERROR_NOT_EQUAL(node.id(),
                          "Stride number of dimensions",
                          stride.raw.size(),
                          "output number of dimensions",
                          output_inst.size.raw.size(),
                          "stride/output dims mismatch");

    auto split = node.get_split();
    for (decltype(split) j = 0; j < split; j++) {
        auto filter_inst = node.weights(j).get_output_layout();  // convolution filter

        auto input_offset = argument.input_offset;

        CLDNN_ERROR_NOT_EQUAL(node.id(),
                              "Weights number of dimensions",
                              filter_inst.size.raw.size(),
                              "output number of dimensions",
                              output_inst.size.raw.size(),
                              "Weights/output dims mismatch");
        CLDNN_ERROR_NOT_EQUAL(node.id(),
                              "Convolution padding mode",
                              node.get_output_layout().data_padding.filling_value(),
                              "padding value",
                              0.0f,
                              "Unknown padding mode.");
        CLDNN_ERROR_NOT_EQUAL(node.id(),
                              "Input offset number of dimensions",
                              input_offset.raw.size(),
                              "input number of dimensions",
                              input_inst.size.raw.size(),
                              "Input offset/ input size mismatch");
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
                              (input_inst.size.feature[0] - input_offset.feature[0]) / split,
                              "input feature maps number",
                              filter_inst.size.feature[0],
                              "Weights/ifm mismatch");
    }
}
}  // namespace cldnn
