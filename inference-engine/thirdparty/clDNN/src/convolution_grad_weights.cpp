/*
// Copyright (c) 2018 Intel Corporation
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
#include "convolution_grad_weights_inst.h"
#include "primitive_type_base.h"
#include "sliding_window_utils.h"
#include "error_handler.h"
#include "json_object.h"
#include <string>

namespace cldnn {
primitive_type_id convolution_grad_weights_type_id() {
    static primitive_type_base<convolution_grad_weights> instance;
    return &instance;
}

layout convolution_grad_weights_inst::calc_output_layout(convolution_grad_weights_node const& node) {
    assert(static_cast<bool>(node.get_primitive()->output_data_type) == false &&
           "Output data type forcing is not supported for "
           "convolution_grad_weights_node!");
    // output buffer will not be used in this primitive unless output gradient weights is set
    auto input_grad_layout_size = node.input(0).get_output_layout();
    tensor output_sizes = {1, 1, 1, 1};
    if (node.output_grad_w())
        output_sizes = node.weights().get_output_layout().size;

    return {input_grad_layout_size.data_type, input_grad_layout_size.format, output_sizes};
}

std::string convolution_grad_weights_inst::to_string(convolution_grad_weights_node const& node) {
    auto desc = node.get_primitive();
    auto strd = desc->stride;
    auto dilation = desc->dilation;
    auto split = desc->split();
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;
    std::stringstream ss_weights, ss_biases;

    for (size_t i = 0; i < desc->weights.size(); ++i) {
        ss_weights << node.weights(i).id();
        ss_weights << ", count: " << node.weights(i).get_output_layout().count();
        i != (desc->weights.size() - 1) ? ss_weights << ", " : ss_weights << "";
        if (node.get_depthwise_sep_opt())
            break;
    }

    for (size_t i = 0; i < desc->bias.size(); ++i) {
        ss_biases << node.bias(i).id();
        ss_biases << ", count: " << node.bias(i).get_output_layout().count();
        i != (desc->bias.size() - 1) ? ss_biases << ", " : ss_biases << "";
    }

    json_composite deconv_info;
    deconv_info.add("weights count", desc->weights.size());
    deconv_info.add("bias count", desc->bias.size());
    deconv_info.add("stride", strd.to_string());
    deconv_info.add("input offset", desc->input_offset.to_string());
    deconv_info.add("dilation", dilation.to_string());
    deconv_info.add("split", split);

    node_info->add("convolution_grad_weights info", deconv_info);
    node_info->dump(primitive_description);
    return primitive_description.str();
}

convolution_grad_weights_inst::typed_primitive_inst(network_impl& network, convolution_grad_weights_node const& node)
    : parent(network, node) {
    auto stride = argument.stride;
    auto dilation = argument.dilation;

    auto input_inst = node.input(1).get_output_layout();
    auto input_grad_inst = node.input().get_output_layout();
    auto desc = node.get_primitive();
    auto output_inst = node.get_output_layout();
    auto output_size = output_inst.size;

    CLDNN_ERROR_NOT_EQUAL(node.id(),
                          "convolution_grad_weights Input_grad size",
                          input_grad_inst.size.raw.size(),
                          "Input size",
                          output_inst.size.raw.size(),
                          "Input_grad/Input number of dimension does not match.");
    CLDNN_ERROR_NOT_EQUAL(node.id(),
                          "convolution_grad_weights Input size",
                          input_inst.size.raw.size(),
                          "output size",
                          output_inst.size.raw.size(),
                          "Input/output number of dimension does not match.");
    CLDNN_ERROR_NOT_EQUAL(node.id(),
                          "convolution_grad_weights Stride size",
                          stride.raw.size(),
                          "output size",
                          output_inst.size.raw.size(),
                          "Stride/output number of dimension does not match.");

    // TODO: add support to dilation not equal 1, 1
    CLDNN_ERROR_NOT_EQUAL(node.id(),
                          "convolution_grad_weights dilation x",
                          dilation.spatial[0],
                          "should be 1",
                          1,
                          "Only dilation x = 1 is supported right now.");
    CLDNN_ERROR_NOT_EQUAL(node.id(),
                          "convolution_grad_weights dilation y",
                          dilation.spatial[1],
                          "should be 1",
                          1,
                          "Only dilation y = 1 is supported right now.");

    if (use_momentum()) {
        CLDNN_ERROR_NOT_EQUAL(node.id(),
                              "number of weights",
                              desc->weights.size(),
                              "should be same as prev_weights_grad number",
                              desc->prev_weights_grad.size(),
                              "");
        if (bias_term())
            CLDNN_ERROR_NOT_EQUAL(node.id(),
                                  "number of bias",
                                  desc->bias.size(),
                                  "should be same as prev_bias_grad number",
                                  desc->prev_bias_grad.size(),
                                  "");
    }

    auto split = node.get_split();
    for (decltype(split) j = 0; j < split; j++) {
        auto& filter_mem = node.weights(j);
        auto filter_inst = filter_mem.get_output_layout();  // convolution_grad_weights filter
        auto input_offset = argument.input_offset;

        if (argument.bias.size() != 0) {
            auto bias_inst = node.bias(j).get_output_layout();
            CLDNN_ERROR_NOT_EQUAL(node.id(),
                                  "Bias batch[0]",
                                  bias_inst.size.batch[0],
                                  "dimension size",
                                  1,
                                  "Batch[0] of bias should be 1. Bias isn't 1D vector.");
            CLDNN_ERROR_NOT_EQUAL(node.id(),
                                  "Bias feature[0]",
                                  bias_inst.size.feature[0],
                                  "dimension size",
                                  1,
                                  "Feature[0] of bias should be 1. Bias isn't 1D vector.");
            CLDNN_ERROR_NOT_EQUAL(node.id(),
                                  "Bias spatial[1]",
                                  bias_inst.size.spatial[1],
                                  "dimension size",
                                  1,
                                  "Spatial[1] of bias should be 1. Bias isn't 1D vector.");

            CLDNN_ERROR_NOT_EQUAL(node.id(),
                                  "Bias spatial[0]",
                                  bias_inst.size.spatial[0],
                                  "input_grad feature size / split",
                                  input_grad_inst.size.feature[0] / split,
                                  "Biases/output feature maps number does not match.");
        }
        CLDNN_ERROR_NOT_EQUAL(node.id(),
                              "convolution_grad_weights padding filling value",
                              node.get_output_layout().data_padding.filling_value(),
                              "padding mode",
                              0.0f,
                              "Unknown padding mode in convolution_grad_weights.");
        CLDNN_ERROR_NOT_EQUAL(node.id(),
                              "Input offset size",
                              input_offset.raw.size(),
                              "input number of dimensions",
                              input_inst.size.raw.size(),
                              "");
        CLDNN_ERROR_NOT_EQUAL(node.id(),
                              "Output feature size",
                              output_size.feature.size(),
                              "expected output feature size",
                              1,
                              "Only one-dimensional features are supported");
        CLDNN_ERROR_NOT_EQUAL(node.id(),
                              "Output feature size",
                              output_size.feature.size(),
                              "expected output feature size",
                              1,
                              "Only one-dimensional features are supported");
        CLDNN_ERROR_NOT_EQUAL(node.id(),
                              "Output batch size",
                              output_size.batch.size(),
                              "expected output batch size",
                              1,
                              "Only one-dimensional features are supported");

        CLDNN_ERROR_LESS_THAN(node.id(),
                              "Weights feature maps number",
                              (input_grad_inst.size.feature[0] - input_offset.feature[0]) / split,
                              "input_grad feature maps number",
                              filter_inst.size.batch[0],
                              "Weights/ifm mimsmatch");
    }
}
}  // namespace cldnn
