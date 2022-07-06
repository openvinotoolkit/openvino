// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "pass_manager.h"
#include "convolution_inst.h"
#include "primitive_type_base.h"
#include "sliding_window_utils.hpp"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include <string>

using namespace ov::runtime::intel_gpu;

namespace cldnn {
primitive_type_id convolution::type_id() {
    static primitive_type_base<convolution> instance;
    return &instance;
}

layout convolution_inst::calc_output_layout(convolution_node const& node) {
    auto desc = node.get_primitive();

    auto input_layout = node.input().get_output_layout();
    auto weights_layout = node.weights(0).get_output_layout().convert_to_weights_layout(desc->grouped_weights_shape);

    auto pad = desc->pad;
    auto stride = desc->stride;
    auto dilation = desc->dilation;
    auto split = desc->weights.size();

    // compute how many outputs in rows and columns will be generate by filter.
    // outp <= (input_size + (2*pad) - kernel_size)/ stride
    auto filter_size = weights_layout.size;

    auto input_type = input_layout.data_type;

    auto output_type = input_type;
    if (node.has_fused_primitives()) {
        output_type = node.get_fused_output_layout().data_type;
    }

    if ((input_type == data_types::u8 || input_type == data_types::i8) &&
         !node.has_fused_primitives()) {
        output_type = data_types::f32;
    }

    uint32_t stride_z = stride.size() >= 3 ? stride[stride.size() - 3] : 1;
    uint32_t stride_y = stride.size() >= 2 ? stride[stride.size() - 2] : 1;
    uint32_t stride_x = stride.size() >= 1 ? stride[stride.size() - 1] : 1;

    uint32_t dilation_z = dilation.size() >= 3 ? dilation[dilation.size() - 3] : 1;
    uint32_t dilation_y = dilation.size() >= 2 ? dilation[dilation.size() - 2] : 1;
    uint32_t dilation_x = dilation.size() >= 1 ? dilation[dilation.size() - 1] : 1;

    // TODO: Consider moving general parameter verification to arguments constructor.
    CLDNN_ERROR_LESS_OR_EQUAL_THAN(node.id(),
                                   "Stride spatial X",
                                   stride_x,
                                   "value",
                                   0,
                                   "Stride spatial X must be positive (>= 1)");
    CLDNN_ERROR_LESS_OR_EQUAL_THAN(node.id(),
                                   "Stride spatial Y",
                                   stride_y,
                                   "value",
                                   0,
                                   "Stride spatial Y must be positive (>= 1)");
    CLDNN_ERROR_LESS_OR_EQUAL_THAN(node.id(),
                                   "Dilatation spatial X",
                                   dilation_x,
                                   "value",
                                   0,
                                   "Dilatation patial X must be positive (>= 1)");
    CLDNN_ERROR_LESS_OR_EQUAL_THAN(node.id(),
                                   "Dilatation spatial Y",
                                   dilation_y,
                                   "value",
                                   0,
                                   "Dilatation spatial Y must be positive (>= 1)");

    if (input_layout.format.spatial_num() == 3) {
        // convolution 3D
        CLDNN_ERROR_LESS_OR_EQUAL_THAN(node.id(),
                                       "Stride spatial Z",
                                       stride_z,
                                       "value",
                                       0,
                                       "Stride spatial Z must be positive (>= 1)");
        CLDNN_ERROR_LESS_OR_EQUAL_THAN(node.id(),
                                       "Dilatation spatial Z",
                                       dilation_z,
                                       "value",
                                       0,
                                       "Dilatation spatial Z must be positive (>= 1)");
    }

    if (input_layout.format == format::winograd_2x3_s1_weights ||
        input_layout.format == format::winograd_2x3_s1_fused_weights ||
        input_layout.format == format::winograd_6x3_s1_fused_weights ||
        input_layout.format == format::image_2d_weights_winograd_6x3_s1_fbxyb ||
        input_layout.format == format::image_2d_weights_winograd_6x3_s1_xfbyb)
        CLDNN_ERROR_MESSAGE(
            node.id(),
            "Input for convolution should not be in winograd weights format - it is reserved for weights only");

    if (input_layout.format == format::winograd_2x3_s1_data) {
        CLDNN_ERROR_NOT_EQUAL(node.id(),
                              "convolution split",
                              split,
                              "expected value",
                              1,
                              "Convolution with winograd input only supports split == 1");
        CLDNN_ERROR_NOT_EQUAL(node.id(),
                              "stride spatial X",
                              stride_x,
                              "expected value",
                              1,
                              "Convolution's input in winograd_2x3_s1_data format can only be used with stride 1x1");
        CLDNN_ERROR_NOT_EQUAL(node.id(),
                              "stride spatial Y",
                              stride_y,
                              "expected value",
                              1,
                              "Convolution's input in winograd_2x3_s1_data format can only be used with stride 1x1");
        CLDNN_ERROR_NOT_EQUAL(node.id(),
                              "Dilatation spatial X",
                              dilation_x,
                              "expected value",
                              1,
                              "Winograd 2x3 convolution does not support dilatation");
        CLDNN_ERROR_NOT_EQUAL(node.id(),
                              "Dilatation spatial Y",
                              dilation_y,
                              "expected value",
                              1,
                              "Winograd 2x3 convolution does not support dilatation");
        if (input_layout.feature() % 32 != 0)
            CLDNN_ERROR_MESSAGE(node.id(),
                                "Input for winograd 2x3 convolution should have features count divisable by 32");
        if (weights_layout.ofm() % 32 != 0)
            CLDNN_ERROR_MESSAGE(node.id(),
                                "Number of filters (OFM) for winograd 2x3 convolution should be divisable by 32");

        CLDNN_ERROR_LESS_THAN(node.id(),
                              "input width",
                              input_layout.spatial(0),
                              "filter width",
                              3,
                              "Convolution input is smaller than weights");
        CLDNN_ERROR_LESS_THAN(node.id(),
                              "input height",
                              input_layout.spatial(1),
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
                      tensor{input_layout.batch(),
                             weights_layout.ofm() * weights_layout.group(),
                             input_layout.spatial(0),
                             input_layout.spatial(1) - winograd_filter_height + 1},
                      input_layout.data_padding};
    }

    // Adjust output format for mixed precision case in onednn
    auto out_fmt = input_layout.format;
    bool is_2d = (input_layout.format.spatial_num() == 2);
    bool is_3d = (input_layout.format.spatial_num() == 3);
    if (node.get_preferred_impl_type() == impl_types::onednn) {
        #if 1
        if (data_type_traits::is_i8_u8(output_type)) {
            if (is_2d) {
                if (input_layout.format == format::b_fs_yx_fsv16)
                    out_fmt = format::b_fs_yx_fsv32;
                else if (input_layout.format == format::bs_fs_yx_bsv32_fsv16)
                    out_fmt = format::bs_fs_yx_bsv32_fsv32;
                else if (input_layout.format == format::b_fs_yx_fsv2)
                    out_fmt = format::b_fs_yx_fsv32;
            } else if (is_3d) {
                if (input_layout.format == format::b_fs_zyx_fsv16)
                    out_fmt = format::b_fs_zyx_fsv32;
                else if (input_layout.format == format::bs_fs_zyx_bsv32_fsv16)
                    out_fmt = format::bs_fs_zyx_bsv32_fsv32;
            }
        } else if (data_type_traits::is_floating_point(output_type)) {
            if (is_2d) {
                if (input_layout.format == format::b_fs_yx_fsv32)
                    out_fmt = format::b_fs_yx_fsv16;
                else if (input_layout.format == format::bs_fs_yx_bsv32_fsv32)
                    out_fmt = format::bs_fs_yx_bsv32_fsv16;
            } else if (is_3d) {
                if (input_layout.format == format::b_fs_zyx_fsv32)
                    out_fmt = format::b_fs_zyx_fsv16;
                else if (input_layout.format == format::bs_fs_zyx_bsv32_fsv32)
                    out_fmt = format::bs_fs_zyx_bsv32_fsv16;
                else if (input_layout.format == format::b_fs_zyx_fsv2)
                    out_fmt = format::b_fs_zyx_fsv16;
                else if (input_layout.format == format::bs_fs_zyx_bsv8_fsv2)
                    out_fmt = input_layout.batch() > 16 ? format::bs_fs_zyx_bsv32_fsv16 : format::b_fs_zyx_fsv16;
            }
        }
        #endif

        out_fmt = node.get_required_output();
    }

    // get output feature map from weights. It should be the same as number of biases. Will be verifed in
    // convolution::create()
    auto group = desc->groups;
    int32_t number_of_features = 0;
    if (desc->grouped_weights_shape && !format::is_grouped(weights_layout.format)) {
        number_of_features = weights_layout.size.feature[0] * static_cast<int32_t>(group);
    } else {
        if (format::is_grouped(weights_layout.format)) {
            number_of_features = weights_layout.size.batch[0] * static_cast<int32_t>(group);
        } else {
            number_of_features = weights_layout.size.batch[0];
        }
    }

    if (desc->with_output_size) {
        CLDNN_ERROR_LESS_OR_EQUAL_THAN(node.id(),
                                       "User defined output spatial X",
                                       desc->output_size.spatial[0],
                                       "value",
                                       0,
                                       "must be positive(>= 1)");
        CLDNN_ERROR_LESS_OR_EQUAL_THAN(node.id(),
                                       "User defined output spatial Y",
                                       desc->output_size.spatial[1],
                                       "value",
                                       0,
                                       "must be positive(>= 1)");
        CLDNN_ERROR_LESS_OR_EQUAL_THAN(node.id(),
                                       "User defined output spatial Z",
                                       desc->output_size.spatial[2],
                                       "value",
                                       0,
                                       "must be positive(>= 1)");

        tensor output_size(input_layout.batch(),
                           desc->output_size.feature[0],
                           desc->output_size.spatial[0],
                           desc->output_size.spatial[1],
                           desc->output_size.spatial[2]);
        if (output_type == data_types::bin) {
            return {output_type, format::b_fs_yx_32fp, output_size};
        }

        return {output_type, out_fmt, output_size};
    }

    auto output_range = calc_sliding_window_output_range<swor_mode::all>(input_layout.size,
                                                                         filter_size,
                                                                         pad,
                                                                         stride,
                                                                         dilation,
                                                                         true,
                                                                         1);

    tensor output_size = tensor(input_layout.batch(),
                                weights_layout.ofm() * weights_layout.group(),
                                output_range.spatial[0],
                                output_range.spatial[1],
                                output_range.spatial[2]);


    if (output_type == data_types::bin) {
        return {output_type, format::b_fs_yx_32fp, output_size};
    }
    return {output_type, out_fmt, output_size};
}

std::string convolution_inst::to_string(convolution_node const& node) {
    auto desc = node.get_primitive();
    auto strd = desc->stride;
    auto split = node.get_split();
    auto groups = node.get_groups();
    auto dilation = desc->dilation;
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;

    std::string w_zp = desc->weights_zero_points.empty() ? "false" : "true";
    std::string a_zp = desc->activations_zero_points.empty() ? "false" : "true";

    json_composite conv_info;
    conv_info.add("stride", cldnn::to_string(strd));
    conv_info.add("pad", cldnn::to_string(desc->pad));
    conv_info.add("padding above", cldnn::to_string(desc->padding_above));
    conv_info.add("padding below", cldnn::to_string(desc->padding_below));
    conv_info.add("split", split);
    conv_info.add("groups", groups);
    conv_info.add("dilation", cldnn::to_string(dilation));
    conv_info.add("deformable_groups", desc->deformable_groups);
    conv_info.add("groups", desc->groups);
    conv_info.add("has zero points for weights: ", w_zp);
    conv_info.add("has zero points for activations: ", a_zp);

    if (desc->with_output_size) {
        json_composite ud_out_size_info;
        ud_out_size_info.add("size", desc->output_size.to_string());
        conv_info.add("with user defined output size", ud_out_size_info);
    }

    node_info->add("convolution info", conv_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

convolution_inst::typed_primitive_inst(network& network, convolution_node const& node) : parent(network, node) {
    auto stride = argument.stride;
    auto pad = argument.pad;

    auto input_layout = node.input().get_output_layout();
    auto output_layout = node.get_output_layout();
    auto output_size = output_layout.size;

    CLDNN_ERROR_NOT_EQUAL(node.id(),
                          "Input number of dimensions",
                          input_layout.get_rank(),
                          "output number of dimensions",
                          output_layout.get_rank(),
                          "Input/output rank mismatch");

    auto split = node.get_split();
    for (decltype(split) j = 0; j < split; j++) {
        auto filter_inst = node.weights(j).get_output_layout().convert_to_weights_layout(argument.grouped_weights_shape);

        if (bias_term()) {
            auto bias_inst = node.bias(j).get_output_layout();
            CLDNN_ERROR_NOT_EQUAL(node.id(),
                                  "Bias batch[0]",
                                  bias_inst.batch(),
                                  "expected size of batch",
                                  1,
                                  "Biases isn't 1D vector.");
            CLDNN_ERROR_NOT_EQUAL(node.id(),
                                  "Bias feature[0]",
                                  bias_inst.feature(),
                                  "expected feature map number",
                                  output_size.feature[0] / split,
                                  "Bias/fm mismatch");
            CLDNN_ERROR_NOT_EQUAL(node.id(),
                                  "Bias spatial[2]",
                                  bias_inst.spatial(2),
                                  "expected size of spatial[2]",
                                  1,
                                  "Biases isn't 1D vector.");
            CLDNN_ERROR_NOT_EQUAL(node.id(),
                                  "Bias spatial[1]",
                                  bias_inst.spatial(1),
                                  "expected size of spatial[1]",
                                  1,
                                  "Biases isn't 1D vector.");
            CLDNN_ERROR_NOT_EQUAL(node.id(),
                                  "Bias spatial[0]",
                                  bias_inst.spatial(0),
                                  "expected size of spatial[0]",
                                  1,
                                  "Biases isn't 1D vector.");
        }

        auto pad = argument.pad;

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
        CLDNN_ERROR_NOT_EQUAL(node.id(),
                              "Weights feature maps number",
                              filter_inst.ifm() * filter_inst.group(),
                              "input feature maps number",
                              input_layout.feature(),
                              "Weights/ifm mismatch");
    }
}
}  // namespace cldnn
