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
#include "fused_conv_eltwise_inst.h"
#include "primitive_type_base.h"
#include "sliding_window_utils.h"
#include "error_handler.h"
#include "json_object.h"
#include <string>

namespace cldnn {
primitive_type_id fused_conv_eltwise_type_id() {
    static primitive_type_base<fused_conv_eltwise> instance;
    return &instance;
}

layout fused_conv_eltwise_inst::calc_output_layout(fused_conv_eltwise_node const& node) {
    auto desc = node.get_primitive();

    auto input_layout = node.input().get_output_layout();
    auto weights_layout = node.weights(0).get_output_layout();  // weights are stored after inputs

    auto input_offset = desc->conv.input_offset;
    auto stride = desc->conv.stride;
    auto dilation = desc->conv.dilation;
    auto split = desc->conv.weights.size();

    // compute how many outputs in rows and columns will be generate by filter.
    // outp <= (input_size - (2*input_offset) - kernel_size)/ stride
    auto filter_size = weights_layout.size;

    auto input_type = input_layout.data_type;
    auto output_type = node.get_primitive()->output_data_type ? *node.get_primitive()->output_data_type : input_type;

    // TODO: Consider moving general parameter verification to arguments constructor.
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

    // TODO: FCN and SSD used offset larger than convolution size. does it make sense to support it? do we support it on
    // the ref kernels?
    //     CLDNN_ERROR_GREATER_THAN(node.id(), "Negate input offset spatial X", -input_offset.spatial[0], "input window
    //     size spatial X", filter_size.spatial[0], "First convolution is outside of image. please reduce input offset
    //     X"); CLDNN_ERROR_GREATER_THAN(node.id(), "Negate input offset spatial Y", -input_offset.spatial[1], "input
    //     window size spatial Y", filter_size.spatial[1], "First convolution is outside of image. please reduce input
    //     offset Y");

    if (input_layout.format == format::winograd_2x3_s1_weights ||
        input_layout.format == format::winograd_2x3_s1_fused_weights ||
        input_layout.format == format::winograd_6x3_s1_fused_weights ||
        input_layout.format == format::image_2d_weights_winograd_6x3_s1_fbxyb ||
        input_layout.format == format::image_2d_weights_winograd_6x3_s1_xfbyb)
        CLDNN_ERROR_MESSAGE(
            node.id(),
            "Input for convolution should not be in windograd weights format - it is reserved for weights only");

    if (input_layout.format == format::winograd_2x3_s1_data) {
        CLDNN_ERROR_NOT_EQUAL(node.id(),
                              "convolution split",
                              split,
                              "expected value",
                              1,
                              "Convolution with winograd input only supports split == 1");
        CLDNN_ERROR_NOT_EQUAL(node.id(),
                              "stride spatial X",
                              stride.spatial[0],
                              "expected value",
                              1,
                              "Convolution's input in winograd_2x3_s1_data format can only be used with stride 1x1");
        CLDNN_ERROR_NOT_EQUAL(node.id(),
                              "stride spatial Y",
                              stride.spatial[1],
                              "expected value",
                              1,
                              "Convolution's input in winograd_2x3_s1_data format can only be used with stride 1x1");
        CLDNN_ERROR_NOT_EQUAL(node.id(),
                              "Dilatation spatial X",
                              dilation.spatial[0],
                              "expected value",
                              1,
                              "Winograd 2x3 convolution does not support dilatation");
        CLDNN_ERROR_NOT_EQUAL(node.id(),
                              "Dilatation spatial Y",
                              dilation.spatial[1],
                              "expected value",
                              1,
                              "Winograd 2x3 convolution does not support dilatation");
        if (input_layout.size.feature[0] % 32 != 0)
            CLDNN_ERROR_MESSAGE(node.id(),
                                "Input for winograd 2x3 convolution should have features count divisable by 32");
        if (weights_layout.size.batch[0] % 32 != 0)
            CLDNN_ERROR_MESSAGE(node.id(),
                                "Number of filters (OFM) for winograd 2x3 convolution should be divisable by 32");

        if (node.get_primitive()->conv.with_activation)
            CLDNN_ERROR_MESSAGE(node.id(),
                                "Winograd 2x3 convolution should not have activation fused - activation should be "
                                "performed at transformation from winograd domain stage");

        CLDNN_ERROR_LESS_THAN(node.id(),
                              "input width",
                              input_layout.size.spatial[0],
                              "filter width",
                              3,
                              "Convolution input is smaller than weights");
        CLDNN_ERROR_LESS_THAN(node.id(),
                              "input height",
                              input_layout.size.spatial[1],
                              "filter height",
                              3,
                              "Convolution input is smaller than weights");

        constexpr tensor::value_type filter_height =
            3;  // by definition of format::winograd_2x3_s1_data (our assumption)
        constexpr tensor::value_type winograd_filter_height =
            filter_height;  // for this format, winograd filter is considered to be a set of 1d filters so its height
                            // should remain the same as original filter's

        return layout{output_type,
                      input_layout.format,
                      tensor{input_layout.size.batch[0],
                             weights_layout.size.batch[0],
                             input_layout.size.spatial[0],
                             input_layout.size.spatial[1] - winograd_filter_height + 1},
                      input_layout.data_padding};
    }

    // get output feature map from weights. It should be the same as number of biases. Will be verifed in
    // convolution::create()
    auto number_of_features = weights_layout.size.batch[0] * static_cast<int32_t>(split);

    if (desc->conv.with_output_size) {
        CLDNN_ERROR_LESS_OR_EQUAL_THAN(node.id(),
                                       "User defined output spatial X",
                                       desc->conv.output_size.spatial[0],
                                       "value",
                                       0,
                                       "must be positive(>= 1)");
        CLDNN_ERROR_LESS_OR_EQUAL_THAN(node.id(),
                                       "User defined output spatial Y",
                                       desc->conv.output_size.spatial[1],
                                       "value",
                                       0,
                                       "must be positive(>= 1)");

        tensor output_size(input_layout.size.batch[0],
                           number_of_features,
                           desc->conv.output_size.spatial[0],
                           desc->conv.output_size.spatial[1]);
        return {output_type, input_layout.format, output_size};
    }

    auto output_range = calc_sliding_window_output_range<swor_mode::all>(input_layout.size,
                                                                         filter_size,
                                                                         input_offset,
                                                                         stride,
                                                                         dilation,
                                                                         true,
                                                                         1);

    tensor output_size(input_layout.size.batch[0],
                       number_of_features,
                       output_range.spatial[0],
                       output_range.spatial[1]);

    // due to performance reason for using fs_bs_yx_bsv4_fsv32 first convolution have 3 features, so first conv layer
    // will take byxf and return fs_bs_yx_bsv4_fsv32
    if (input_layout.data_type == data_types::i8 && input_layout.format == format::byx8_f4 &&
        input_layout.size.batch[0] % 4 == 0 && input_layout.size.feature[0] == 3) {
        return layout{output_type, cldnn::format::fs_bs_yx_bsv4_fsv32, output_size};
    }

    return {output_type, input_layout.format, output_size};
}

std::string fused_conv_eltwise_inst::to_string(fused_conv_eltwise_node const& node) {
    auto desc = node.get_primitive();
    auto strd = desc->conv.stride;
    auto split = node.get_split();
    auto dilation = desc->conv.dilation;
    auto node_info = node.desc_to_json();
    auto activation = desc->conv.with_activation ? " true" : "false";

    std::stringstream primitive_description;

    json_composite conv_info;
    conv_info.add("stride", strd.to_string());
    conv_info.add("input offset", desc->conv.input_offset.to_string());
    conv_info.add("split", split);
    conv_info.add("dilation", dilation.to_string());
    conv_info.add("with activation", activation);
    conv_info.add("slope", desc->conv.activation_negative_slope);
    if (desc->conv.with_output_size) {
        json_composite ud_out_size_info;
        ud_out_size_info.add("size", desc->conv.output_size.to_string());
        conv_info.add("with user defined output size", ud_out_size_info);
    }

    node_info->add("convolution info", conv_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

fused_conv_eltwise_inst::typed_primitive_inst(network_impl& network, fused_conv_eltwise_node const& node)
    : parent(network, node) {
    auto stride = argument.conv.stride;

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
                                  1,
                                  "Biases isn't 1D vector.");
            CLDNN_ERROR_NOT_EQUAL(node.id(),
                                  "Bias spatial[1]",
                                  bias_inst.size.spatial[1],
                                  "expected size of spatial[1]",
                                  1,
                                  "Biases isn't 1D vector.");

            CLDNN_ERROR_NOT_EQUAL(node.id(),
                                  "Bias spatial[0]",
                                  bias_inst.size.spatial[0],
                                  "expected feature map number",
                                  output_size.feature[0] / split,
                                  "Bias/fm mismatch");
        }

        auto input_offset = argument.conv.input_offset;

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
        if (filter_inst.format == format::bf_lyx_yx) {  // local convolution
            auto local = filter_inst.size.local;
            CLDNN_ERROR_NOT_EQUAL(node.id(),
                                  "Number of local x dimension",
                                  local[0],
                                  "output x dimension",
                                  output_inst.size.spatial[0],
                                  "Weights/output dims mismatch");
            CLDNN_ERROR_NOT_EQUAL(node.id(),
                                  "Number of local y dimension",
                                  local[1],
                                  "output y dimension",
                                  output_inst.size.spatial[1],
                                  "Weights/output dims mismatch");
        }
    }
}
}  // namespace cldnn
