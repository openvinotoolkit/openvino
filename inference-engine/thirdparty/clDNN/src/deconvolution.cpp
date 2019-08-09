/*
// Copyright (c) 2016 Intel Corporation
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
#include "deconvolution_inst.h"
#include "primitive_type_base.h"
#include "sliding_window_utils.h"
#include "error_handler.h"
#include "json_object.h"
#include <string>

namespace cldnn {
primitive_type_id deconvolution_type_id() {
    static primitive_type_base<deconvolution> instance;
    return &instance;
}

layout deconvolution_inst::calc_output_layout(deconvolution_node const& node) {
    assert(static_cast<bool>(node.get_primitive()->output_data_type) == false &&
           "Output data type forcing is not supported for deconvolution_node!");
    auto desc = node.get_primitive();

    auto input_layout = node.input().get_output_layout();
    auto weights_layout = node.weights(0).get_output_layout();  // weights are stored after inputs

    auto input_offset = desc->input_offset;
    auto strd = desc->stride;
    auto split = desc->weights.size();

    auto number_of_features = weights_layout.size.batch[0] * static_cast<int32_t>(split);

    // Deconvolution is used for convolution backward pass, but number of features will differ then
    if (desc->gradient())
        number_of_features = weights_layout.size.feature[0] * static_cast<int32_t>(split);

    if (desc->with_output_size) {
        CLDNN_ERROR_LESS_OR_EQUAL_THAN(node.id(),
                                       "User-defined output spatial X",
                                       desc->output_size.spatial[0],
                                       "value 0",
                                       0,
                                       "User-defined size of output layout must be positive (>= 1)");
        CLDNN_ERROR_LESS_OR_EQUAL_THAN(node.id(),
                                       "User-defined output spatial Y",
                                       desc->output_size.spatial[1],
                                       "value 0",
                                       0,
                                       "User-defined size of output layout must be positive (>= 1)");
        CLDNN_ERROR_LESS_OR_EQUAL_THAN(node.id(),
                                       "User-defined output spatial Z",
                                       desc->output_size.spatial[2],
                                       "value 0",
                                       0,
                                       "User-defined size of output layout must be positive (>= 1)");

        tensor output_size(desc->output_size.batch[0],
                           number_of_features,
                           desc->output_size.spatial[0],
                           desc->output_size.spatial[1],
                           desc->output_size.spatial[2]);
        return {input_layout.data_type, input_layout.format, output_size};
    }

    // compute output_dim <= stride * (input_size - 1) + kernel_size + 2 * input_offset;
    auto filter_size = weights_layout.size;

    int32_t off_factor = 2;
    int32_t spatial_dims = cldnn::format::traits(input_layout.format).spatial_num;
    CLDNN_ERROR_GREATER_THAN(node.id(),
                                   "number of spatial dimensions",
                                   spatial_dims,
                                   "expected number of dimensions",
                                   3,
                                   "As for now, deconvolutions with more than 3 dimensions are not supported");

    int32_t x = off_factor * input_offset.spatial[0] + (input_layout.size.spatial[0] - 1) * strd.spatial[0] + filter_size.spatial[0];
    int32_t y = 1;
    if (spatial_dims > 1) {
        y = off_factor * input_offset.spatial[1] + (input_layout.size.spatial[1] - 1) * strd.spatial[1] + filter_size.spatial[1];
    }
    int32_t z = 1;
    if (spatial_dims > 2) {
        z = off_factor * input_offset.spatial[2] + (input_layout.size.spatial[2] - 1) * strd.spatial[2] + filter_size.spatial[2];
    }

    tensor output_size(input_layout.size.batch[0],
                       number_of_features, x, y, z);
    return {input_layout.data_type, input_layout.format, output_size};
}

std::string deconvolution_inst::to_string(deconvolution_node const& node) {
    auto desc = node.get_primitive();
    auto strd = desc->stride;
    auto split = desc->split();
    auto node_info = node.desc_to_json();
    auto activation = desc->with_activation ? " true" : "false";

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
        if (node.get_depthwise_sep_opt())
            break;
    }

    json_composite deconv_info;
    deconv_info.add("weights count", desc->weights.size());
    deconv_info.add("bias count", desc->bias.size());
    deconv_info.add("stride", strd.to_string());
    deconv_info.add("input offset", desc->input_offset.to_string());
    deconv_info.add("split", split);
    deconv_info.add("with activation", activation);
    deconv_info.add("slope", desc->activation_negative_slope);
    if (desc->with_output_size) {
        json_composite ud_out_size_info;
        ud_out_size_info.add("size", desc->output_size.to_string());
        deconv_info.add("with_user_defined_output_size", ud_out_size_info);
    }
    node_info->add("deconvolution info", deconv_info);
    node_info->dump(primitive_description);
    return primitive_description.str();
}

deconvolution_inst::typed_primitive_inst(network_impl& network, deconvolution_node const& node)
    : parent(network, node) {
    auto stride = argument.stride;

    auto input_inst = node.input().get_output_layout();
    auto output_inst = node.get_output_layout();
    auto output_size = output_inst.size;

    CLDNN_ERROR_NOT_EQUAL(node.id(),
                          "Input size",
                          input_inst.size.raw.size(),
                          "output size",
                          output_inst.size.raw.size(),
                          "Input/output number of dimension does not match.");
    CLDNN_ERROR_NOT_EQUAL(node.id(),
                          "Stride size",
                          stride.raw.size(),
                          "output size",
                          output_inst.size.raw.size(),
                          "Stride/output number of dimension does not match.");

    auto split = node.get_split();
    for (decltype(split) j = 0; j < split; j++) {
        auto filter_inst = node.weights(j).get_output_layout();  // deconvolution filter
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
                                  "Bias spatial[2]",
                                  bias_inst.size.spatial[1],
                                  "dimension size",
                                  1,
                                  "Spatial[2] of bias should be 1. Bias isn't 1D vector.");

            CLDNN_ERROR_NOT_EQUAL(node.id(),
                                  "Bias spatial[0]",
                                  bias_inst.size.spatial[0],
                                  "output feature size / split",
                                  output_size.feature[0] / split,
                                  "Biases/output feature maps number does not match.");
        }
        CLDNN_ERROR_NOT_EQUAL(node.id(),
                              "deconvolution padding filling value",
                              node.get_output_layout().data_padding.filling_value(),
                              "padding mode",
                              0.0f,
                              "Unknown padding mode in deconvolution.");
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

        if (node.get_primitive()->gradient()) {
            CLDNN_ERROR_LESS_THAN(node.id(),
                                  "Weights feature maps number",
                                  (input_inst.size.feature[0] - input_offset.feature[0]) / split,
                                  "input feature maps number",
                                  filter_inst.size.batch[0],
                                  "Weights/ifm mimsmatch");
        } else {
            CLDNN_ERROR_LESS_THAN(node.id(),
                                  "Weights feature maps number",
                                  (input_inst.size.feature[0] - input_offset.feature[0]) / split,
                                  "input feature maps number",
                                  filter_inst.size.feature[0],
                                  "Weights/ifm mimsmatch");
        }
    }
}
}  // namespace cldnn
