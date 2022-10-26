// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "pass_manager.h"
#include "convolution_inst.h"
#include "primitive_type_base.h"
#include "convolution_shape_inference.hpp"
#include "sliding_window_utils.hpp"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include <string>

using namespace ov::intel_gpu;

namespace cldnn {
primitive_type_id convolution::type_id() {
    static primitive_type_base<convolution> instance;
    return &instance;
}

layout convolution_inst::calc_output_layout(convolution_node const& node, kernel_impl_params const& impl_param) {
    auto desc = impl_param.typed_desc<convolution>();

    auto input_layout = impl_param.get_input_layout();
    auto weights_layout = *impl_param.weights_layout;
    weights_layout = weights_layout.convert_to_weights_layout(desc->grouped_weights_shape);

    auto pad = desc->pad;
    auto stride = desc->stride;
    auto dilation = desc->dilation;
    auto split = desc->weights.size();

    // compute how many outputs in rows and columns will be generate by filter.
    // outp <= (input_size + (2*pad) - kernel_size)/ stride
    auto filter_size = weights_layout.get_tensor();

    auto input_type = input_layout.data_type;

    auto output_type = input_type;
    if (impl_param.has_fused_primitives()) {
        output_type = impl_param.get_fused_output_layout().data_type;
    }

    if ((input_type == data_types::u8 || input_type == data_types::i8) &&
         !impl_param.has_fused_primitives()) {
        output_type = data_types::f32;
    }

    uint32_t stride_z = stride.size() >= 3 ? stride[stride.size() - 3] : 1;
    uint32_t stride_y = stride.size() >= 2 ? stride[stride.size() - 2] : 1;
    uint32_t stride_x = stride.size() >= 1 ? stride[stride.size() - 1] : 1;

    uint32_t dilation_z = dilation.size() >= 3 ? dilation[dilation.size() - 3] : 1;
    uint32_t dilation_y = dilation.size() >= 2 ? dilation[dilation.size() - 2] : 1;
    uint32_t dilation_x = dilation.size() >= 1 ? dilation[dilation.size() - 1] : 1;

    // TODO: Consider moving general parameter verification to arguments constructor.
    CLDNN_ERROR_LESS_OR_EQUAL_THAN(desc->id,
                                   "Stride spatial X",
                                   stride_x,
                                   "value",
                                   0,
                                   "Stride spatial X must be positive (>= 1)");
    CLDNN_ERROR_LESS_OR_EQUAL_THAN(desc->id,
                                   "Stride spatial Y",
                                   stride_y,
                                   "value",
                                   0,
                                   "Stride spatial Y must be positive (>= 1)");
    CLDNN_ERROR_LESS_OR_EQUAL_THAN(desc->id,
                                   "Dilation spatial X",
                                   dilation_x,
                                   "value",
                                   0,
                                   "Dilation patial X must be positive (>= 1)");
    CLDNN_ERROR_LESS_OR_EQUAL_THAN(desc->id,
                                   "Dilation spatial Y",
                                   dilation_y,
                                   "value",
                                   0,
                                   "Dilation spatial Y must be positive (>= 1)");

    if (input_layout.format.spatial_num() == 3) {
        // convolution 3D
        CLDNN_ERROR_LESS_OR_EQUAL_THAN(desc->id,
                                       "Stride spatial Z",
                                       stride_z,
                                       "value",
                                       0,
                                       "Stride spatial Z must be positive (>= 1)");
        CLDNN_ERROR_LESS_OR_EQUAL_THAN(desc->id,
                                       "Dilation spatial Z",
                                       dilation_z,
                                       "value",
                                       0,
                                       "Dilation spatial Z must be positive (>= 1)");
    }

    if (input_layout.format == format::winograd_2x3_s1_weights ||
        input_layout.format == format::winograd_2x3_s1_fused_weights ||
        input_layout.format == format::winograd_6x3_s1_fused_weights ||
        input_layout.format == format::image_2d_weights_winograd_6x3_s1_fbxyb ||
        input_layout.format == format::image_2d_weights_winograd_6x3_s1_xfbyb)
        CLDNN_ERROR_MESSAGE(
            desc->id,
            "Input for convolution should not be in winograd weights format - it is reserved for weights only");

    if (input_layout.format == format::winograd_2x3_s1_data) {
        CLDNN_ERROR_NOT_EQUAL(desc->id,
                              "convolution split",
                              split,
                              "expected value",
                              1,
                              "Convolution with winograd input only supports split == 1");
        CLDNN_ERROR_NOT_EQUAL(desc->id,
                              "stride spatial X",
                              stride_x,
                              "expected value",
                              1,
                              "Convolution's input in winograd_2x3_s1_data format can only be used with stride 1x1");
        CLDNN_ERROR_NOT_EQUAL(desc->id,
                              "stride spatial Y",
                              stride_y,
                              "expected value",
                              1,
                              "Convolution's input in winograd_2x3_s1_data format can only be used with stride 1x1");
        CLDNN_ERROR_NOT_EQUAL(desc->id,
                              "Dilation spatial X",
                              dilation_x,
                              "expected value",
                              1,
                              "Winograd 2x3 convolution does not support dilation");
        CLDNN_ERROR_NOT_EQUAL(desc->id,
                              "Dilation spatial Y",
                              dilation_y,
                              "expected value",
                              1,
                              "Winograd 2x3 convolution does not support dilation");
        if (input_layout.feature() % 32 != 0)
            CLDNN_ERROR_MESSAGE(desc->id,
                                "Input for winograd 2x3 convolution should have features count divisable by 32");
        if (weights_layout.ofm() % 32 != 0)
            CLDNN_ERROR_MESSAGE(desc->id,
                                "Number of filters (OFM) for winograd 2x3 convolution should be divisable by 32");

        CLDNN_ERROR_LESS_THAN(desc->id,
                              "input width",
                              input_layout.spatial(0),
                              "filter width",
                              3,
                              "Convolution input is smaller than weights");
        CLDNN_ERROR_LESS_THAN(desc->id,
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

    // Adjust output format for shallow conv and mixed precision cases in onednn
    auto out_fmt = input_layout.format;
    if (node.get_preferred_impl_type() == impl_types::onednn && node.get_preferred_output_fmt() != format::any) {
        out_fmt = node.get_preferred_output_fmt();
    }

    // get output feature map from weights. It should be the same as number of biases. Will be verifed in
    // convolution::create()
    if (desc->with_output_size) {
        CLDNN_ERROR_LESS_OR_EQUAL_THAN(desc->id,
                                       "User defined output spatial X",
                                       desc->output_size.spatial[0],
                                       "value",
                                       0,
                                       "must be positive(>= 1)");
        CLDNN_ERROR_LESS_OR_EQUAL_THAN(desc->id,
                                       "User defined output spatial Y",
                                       desc->output_size.spatial[1],
                                       "value",
                                       0,
                                       "must be positive(>= 1)");
        CLDNN_ERROR_LESS_OR_EQUAL_THAN(desc->id,
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

    auto output_range = calc_sliding_window_output_range<swor_mode::all>(input_layout.get_tensor(),
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

template<typename ShapeType>
std::vector<layout> convolution_inst::calc_output_layouts(convolution_node const& node, kernel_impl_params const& impl_param) {
    auto desc = impl_param.typed_desc<convolution>();

    auto input_layout = impl_param.get_input_layout(0);
    auto weights_layout = *impl_param.weights_layout;
    weights_layout = weights_layout.convert_to_weights_layout(desc->grouped_weights_shape);

    if (input_layout.is_dynamic())
        return {layout{ShapeType::dynamic(input_layout.get<ShapeType>().rank()), input_layout.data_type, input_layout.format}};

    auto pad = desc->pad;
    auto stride = desc->stride;
    auto dilation = desc->dilation;
    auto split = desc->weights.size();

    // compute how many outputs in rows and columns will be generate by filter.
    // outp <= (input_size + (2*pad) - kernel_size)/ stride
    auto filter_size = weights_layout.get_tensor();

    auto input_type = input_layout.data_type;

    auto output_type = input_type;
    if (impl_param.has_fused_primitives()) {
        output_type = impl_param.get_fused_output_layout().data_type;
    }

    if ((input_type == data_types::u8 || input_type == data_types::i8) &&
         !impl_param.has_fused_primitives()) {
        output_type = data_types::f32;
    }

    uint32_t stride_z = stride.size() >= 3 ? stride[stride.size() - 3] : 1;
    uint32_t stride_y = stride.size() >= 2 ? stride[stride.size() - 2] : 1;
    uint32_t stride_x = stride.size() >= 1 ? stride[stride.size() - 1] : 1;

    uint32_t dilation_z = dilation.size() >= 3 ? dilation[dilation.size() - 3] : 1;
    uint32_t dilation_y = dilation.size() >= 2 ? dilation[dilation.size() - 2] : 1;
    uint32_t dilation_x = dilation.size() >= 1 ? dilation[dilation.size() - 1] : 1;

    // TODO: Consider moving general parameter verification to arguments constructor.
    CLDNN_ERROR_LESS_OR_EQUAL_THAN(desc->id,
                                   "Stride spatial X",
                                   stride_x,
                                   "value",
                                   0,
                                   "Stride spatial X must be positive (>= 1)");
    CLDNN_ERROR_LESS_OR_EQUAL_THAN(desc->id,
                                   "Stride spatial Y",
                                   stride_y,
                                   "value",
                                   0,
                                   "Stride spatial Y must be positive (>= 1)");
    CLDNN_ERROR_LESS_OR_EQUAL_THAN(desc->id,
                                   "Dilation spatial X",
                                   dilation_x,
                                   "value",
                                   0,
                                   "Dilation patial X must be positive (>= 1)");
    CLDNN_ERROR_LESS_OR_EQUAL_THAN(desc->id,
                                   "Dilation spatial Y",
                                   dilation_y,
                                   "value",
                                   0,
                                   "Dilation spatial Y must be positive (>= 1)");

    if (input_layout.format.spatial_num() == 3) {
        // convolution 3D
        CLDNN_ERROR_LESS_OR_EQUAL_THAN(desc->id,
                                       "Stride spatial Z",
                                       stride_z,
                                       "value",
                                       0,
                                       "Stride spatial Z must be positive (>= 1)");
        CLDNN_ERROR_LESS_OR_EQUAL_THAN(desc->id,
                                       "Dilation spatial Z",
                                       dilation_z,
                                       "value",
                                       0,
                                       "Dilation spatial Z must be positive (>= 1)");
    }

    if (input_layout.format == format::winograd_2x3_s1_weights ||
        input_layout.format == format::winograd_2x3_s1_fused_weights ||
        input_layout.format == format::winograd_6x3_s1_fused_weights ||
        input_layout.format == format::image_2d_weights_winograd_6x3_s1_fbxyb ||
        input_layout.format == format::image_2d_weights_winograd_6x3_s1_xfbyb)
        CLDNN_ERROR_MESSAGE(
            desc->id,
            "Input for convolution should not be in winograd weights format - it is reserved for weights only");

    if (input_layout.format == format::winograd_2x3_s1_data) {
        CLDNN_ERROR_NOT_EQUAL(desc->id,
                              "convolution split",
                              split,
                              "expected value",
                              1,
                              "Convolution with winograd input only supports split == 1");
        CLDNN_ERROR_NOT_EQUAL(desc->id,
                              "stride spatial X",
                              stride_x,
                              "expected value",
                              1,
                              "Convolution's input in winograd_2x3_s1_data format can only be used with stride 1x1");
        CLDNN_ERROR_NOT_EQUAL(desc->id,
                              "stride spatial Y",
                              stride_y,
                              "expected value",
                              1,
                              "Convolution's input in winograd_2x3_s1_data format can only be used with stride 1x1");
        CLDNN_ERROR_NOT_EQUAL(desc->id,
                              "Dilation spatial X",
                              dilation_x,
                              "expected value",
                              1,
                              "Winograd 2x3 convolution does not support dilation");
        CLDNN_ERROR_NOT_EQUAL(desc->id,
                              "Dilation spatial Y",
                              dilation_y,
                              "expected value",
                              1,
                              "Winograd 2x3 convolution does not support dilation");
        if (input_layout.feature() % 32 != 0)
            CLDNN_ERROR_MESSAGE(desc->id,
                                "Input for winograd 2x3 convolution should have features count divisable by 32");
        if (weights_layout.ofm() % 32 != 0)
            CLDNN_ERROR_MESSAGE(desc->id,
                                "Number of filters (OFM) for winograd 2x3 convolution should be divisable by 32");

        CLDNN_ERROR_LESS_THAN(desc->id,
                              "input width",
                              input_layout.spatial(0),
                              "filter width",
                              3,
                              "Convolution input is smaller than weights");
        CLDNN_ERROR_LESS_THAN(desc->id,
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

        return {cldnn::layout{output_type,
                      input_layout.format,
                      tensor{input_layout.batch(),
                             weights_layout.ofm() * weights_layout.group(),
                             input_layout.spatial(0),
                             input_layout.spatial(1) - winograd_filter_height + 1},
                      input_layout.data_padding}};
    }

    // Adjust output format for shallow conv and mixed precision cases in onednn
    auto out_fmt = input_layout.format;
    if (node.get_preferred_impl_type() == impl_types::onednn && node.get_preferred_output_fmt() != format::any) {
        out_fmt = node.get_preferred_output_fmt();
    }

    // get output feature map from weights. It should be the same as number of biases. Will be verifed in
    // convolution::create()
    if (desc->with_output_size) {
        // static case
        CLDNN_ERROR_LESS_OR_EQUAL_THAN(desc->id,
                                       "User defined output spatial X",
                                       desc->output_size.spatial[0],
                                       "value",
                                       0,
                                       "must be positive(>= 1)");
        CLDNN_ERROR_LESS_OR_EQUAL_THAN(desc->id,
                                       "User defined output spatial Y",
                                       desc->output_size.spatial[1],
                                       "value",
                                       0,
                                       "must be positive(>= 1)");
        CLDNN_ERROR_LESS_OR_EQUAL_THAN(desc->id,
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
            return {cldnn::layout{output_type, format::b_fs_yx_32fp, output_size}};
        }

        return {cldnn::layout{output_type, out_fmt, output_size}};
    }
    // dynamic case
    std::vector<ShapeType> input_shapes = {
        input_layout.get<ShapeType>(),
        weights_layout.get<ShapeType>()
    };
    std::vector<ShapeType> output_shapes = {ShapeType()};

    if (desc->groups > 1) {
        ov::op::v1::GroupConvolution op;
        op.set_dilations(desc->dilation);
        op.set_strides(desc->stride);
        op.set_auto_pad(ov::op::PadType::EXPLICIT);
        auto pad_begin = desc->padding_above;
        auto pad_end = desc->padding_below;
        if (input_shapes[1].size() == 4 && input_shapes[0].size() == 3) {
            // 3D
            input_shapes[1][3] = input_shapes[1][2];
            input_shapes[1][2] = input_shapes[0][1].get_length()/input_shapes[1][0].get_length();
        }
        ov::op::v1::shape_infer(&op, pad_begin, pad_end, input_shapes, output_shapes);
    } else {
        ov::op::v1::Convolution op;
        op.set_dilations(desc->dilation);
        op.set_strides(desc->stride);
        op.set_auto_pad(ov::op::PadType::EXPLICIT);
        auto pad_begin = desc->padding_above;
        auto pad_end = desc->padding_below;
        ov::op::v1::shape_infer(&op, pad_begin, pad_end, input_shapes, output_shapes);
    }
    format::type output_format = input_layout.format.value;
    return {layout{output_shapes[0], output_type, output_format}};
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

convolution_inst::typed_primitive_inst(network& network, convolution_node const& node) :
    parent(network, node),
    _groups(node.get_groups()),
    _split(node.get_split()),
    _deform_conv_dep_offset(node.get_deform_conv_dep_offset()) {
    if (node.is_dynamic())
        return;
    auto stride = argument->stride;
    auto pad = argument->pad;

    auto input_layout = node.input().get_output_layout();
    auto output_layout = node.get_output_layout();
    auto output_size = output_layout.get_tensor();

    CLDNN_ERROR_NOT_EQUAL(node.id(),
                          "Input number of dimensions",
                          input_layout.get_rank(),
                          "output number of dimensions",
                          output_layout.get_rank(),
                          "Input/output rank mismatch");

    auto split = node.get_split();
    for (decltype(split) j = 0; j < split; j++) {
        auto filter_inst = node.weights(j).get_output_layout().convert_to_weights_layout(argument->grouped_weights_shape);

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

        auto pad = argument->pad;

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
