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
#include "fused_conv_bn_scale_inst.h"
#include "primitive_type_base.h"
#include "sliding_window_utils.h"
#include "error_handler.h"
#include "json_object.h"
#include <string>

namespace cldnn {
primitive_type_id fused_conv_bn_scale::type_id() {
    static primitive_type_base<fused_conv_bn_scale> instance;
    return &instance;
}
// TODO: unify this code with regular convolution.
layout fused_conv_bn_scale_inst::calc_output_layout(fused_conv_bn_scale_node const& node) {
    assert(static_cast<bool>(node.get_primitive()->output_data_type) == false &&
           "Output data type forcing is not supported for "
           "fused_conv_bn_scale_node!");
    auto desc = node.get_primitive();

    auto input_layout = node.input().get_output_layout();
    auto weights_layout = node.weights(0).get_output_layout();  // weights are stored after inputs

    auto input_offset = desc->input_offset;
    auto stride = desc->stride;
    auto split = desc->weights.size();
    auto dilation = desc->dilation;

    // compute how many outputs in rows and columns will be generate by filter.
    // outp <= (input_size - (2*input_offset) - kernel_size)/ stride
    auto filter_size = weights_layout.size;

    CLDNN_ERROR_LESS_OR_EQUAL_THAN(node.id(),
                                   "Stride spatial X",
                                   stride.spatial[0],
                                   "value",
                                   0,
                                   "Stride spatial X must be positive (>= 1)");
    CLDNN_ERROR_LESS_OR_EQUAL_THAN(node.id(),
                                   "Stride spatial Y",
                                   stride.spatial[1],
                                   "value",
                                   0,
                                   "Stride spatial Y must be positive (>= 1)");
    CLDNN_ERROR_LESS_OR_EQUAL_THAN(node.id(),
                                   "Dilatation spatial X",
                                   dilation.spatial[0],
                                   "value",
                                   0,
                                   "Dilatation patial X must be positive (>= 1)");
    CLDNN_ERROR_LESS_OR_EQUAL_THAN(node.id(),
                                   "Dilatation spatial Y",
                                   dilation.spatial[1],
                                   "value",
                                   0,
                                   "Dilatation spatial Y must be positive (>= 1)");
    CLDNN_ERROR_GREATER_THAN(node.id(),
                             "Input offset spatial X",
                             2 * input_offset.spatial[0],
                             "input layout spatial X",
                             input_layout.size.spatial[0],
                             "There is no input data to process");
    CLDNN_ERROR_GREATER_THAN(node.id(),
                             "Input offset spatial Y",
                             2 * input_offset.spatial[1],
                             "input layout spatial Y",
                             input_layout.size.spatial[1],
                             "There is no input data to process");
    CLDNN_ERROR_NOT_EQUAL(node.id(),
                          "Input offset feature",
                          input_offset.feature[0],
                          "",
                          0,
                          "Input offset in feature is not supported");
    CLDNN_ERROR_NOT_EQUAL(node.id(),
                          "Input offset batch",
                          input_offset.batch[0],
                          "",
                          0,
                          "Input offset in batch is not supported");

    // get output feature map from weights. It should be the same as number of biases. Will be verified in
    // convolution::create()
    auto number_of_features = weights_layout.size.batch[0] * static_cast<int32_t>(split);

    auto output_range = calc_sliding_window_output_range<swor_mode::all>(input_layout.size,
                                                                         filter_size,
                                                                         input_offset,
                                                                         stride,
                                                                         {1, 1, 1, 1},
                                                                         true,
                                                                         1);

    tensor output_size(input_layout.size.batch[0],
                       number_of_features,
                       output_range.spatial[0],
                       output_range.spatial[1]);
    return {input_layout.data_type, input_layout.format, output_size};
}

std::string fused_conv_bn_scale_inst::to_string(fused_conv_bn_scale_node const& node) {
    auto desc = node.get_primitive();
    auto strd = desc->stride;
    auto split = node.get_split();
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;

    json_composite fuse_info;
    fuse_info.add("stride", strd.to_string());
    fuse_info.add("input offset", desc->input_offset.to_string());
    fuse_info.add("split", split);

    node_info->add("fused_conv_bn_scale info", fuse_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

fused_conv_bn_scale_inst::typed_primitive_inst(network_impl& network, fused_conv_bn_scale_node const& node)
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
                          "Input/output dims mismtach");
    CLDNN_ERROR_NOT_EQUAL(node.id(),
                          "Stride number of dimensions",
                          stride.raw.size(),
                          "output number of dimensions",
                          output_inst.size.raw.size(),
                          "stride/output dims mismtach");

    auto split = node.get_split();
    for (decltype(split) j = 0; j < split; j++) {
        auto filter_inst = node.weights(j).get_output_layout();  // convolution filter
        if (bias_term()) {
            auto bias_inst = node.bias(j).get_output_layout();
            CLDNN_ERROR_NOT_EQUAL(node.id(),
                                  "Bias batch[0]",
                                  bias_inst.size.batch[0],
                                  "expected size of batch",
                                  1,
                                  "Biases isn't 1D vector.");
            CLDNN_ERROR_NOT_EQUAL(node.id(),
                                  "Bias feature[0]",
                                  bias_inst.size.feature[0],
                                  "expected size of feature",
                                  output_size.feature[0] / split,
                                  "Bias/fm mismtach");
            CLDNN_ERROR_NOT_EQUAL(node.id(),
                                  "Bias spatial[1]",
                                  bias_inst.size.spatial[1],
                                  "expected size of spatial[1]",
                                  1,
                                  "Biases isn't 1D vector.");

            CLDNN_ERROR_NOT_EQUAL(node.id(),
                                  "Bias spatial[0]",
                                  bias_inst.size.spatial[0],
                                  "expected size of spatial[0]",
                                  1,
                                  "Biases isn't 1D vector.");
        }

        auto input_offset = argument.input_offset;

        CLDNN_ERROR_NOT_EQUAL(node.id(),
                              "Weights number of dimensions",
                              filter_inst.size.raw.size(),
                              "output number of dimensions",
                              output_inst.size.raw.size(),
                              "Weights/output dims mismtach");
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
                              "Input offset/ input size mismtach");
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
                              "Weights/ifm mismtach");
    }
}
}  // namespace cldnn
