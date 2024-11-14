// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <string>

#include "convolution_inst.h"
#include "convolution_shape_inference.hpp"
#include "group_convolution_shape_inference.hpp"
#include "deformable_convolution_shape_inference.hpp"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include "pass_manager.h"
#include "primitive_type_base.h"

using namespace ov::intel_gpu;
using namespace cldnn;

namespace {
template<typename T, typename V>
T align_to_spatial_rank(const T param, size_t rank, V fill_value) {
    OPENVINO_ASSERT(param.size() <= rank, "[GPU] Can't align convolution parameters to smaller rank");
    std::vector<V> res(rank, fill_value);
    std::copy_n(param.begin(), param.size(), res.begin());
    return T(res);
}

std::vector<layout> calc_output_layout_impl(convolution_node const& node, kernel_impl_params const& impl_param, bool legacy_flow) {
    auto desc = impl_param.typed_desc<convolution>();

    auto input_layout = impl_param.get_input_layout(0);
    auto input_type = input_layout.data_type;
    auto output_type = (input_type == data_types::u8 || input_type == data_types::i8) ? data_types::f32 : input_type;
    if (impl_param.has_fused_primitives()) {
        output_type = impl_param.get_output_element_type();
    }

    auto weights_layout = *impl_param.weights_layout;
    weights_layout = weights_layout.convert_to_weights_layout(desc->grouped_weights_shape);

    if (input_layout.format == format::winograd_2x3_s1_weights ||
        input_layout.format == format::winograd_2x3_s1_fused_weights ||
        input_layout.format == format::winograd_6x3_s1_fused_weights ||
        input_layout.format == format::image_2d_weights_winograd_6x3_s1_fbxyb ||
        input_layout.format == format::image_2d_weights_winograd_6x3_s1_xfbyb)
        CLDNN_ERROR_MESSAGE(
            desc->id,
            "Input for convolution should not be in winograd weights format - it is reserved for weights only");

    if (input_layout.format == format::winograd_2x3_s1_data) {
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
    auto output_format = input_layout.format;
    if (node.get_preferred_impl_type() == impl_types::onednn && node.get_preferred_output_fmt() != format::any) {
        output_format = node.get_preferred_output_fmt();
    }

    // dynamic case
    std::vector<ov::PartialShape> input_shapes = {
        input_layout.get_partial_shape(),
        weights_layout.get_partial_shape()
    };

    std::vector<ov::PartialShape> output_shapes;

    auto pads_begin = desc->padding_begin;
    auto pads_end = desc->padding_end;
    auto dilation = desc->dilation;
    auto strides = desc->stride;

    if (legacy_flow) {
        auto spatial_rank = impl_param.get_input_layout(0).get_spatial_rank();
        dilation = align_to_spatial_rank(dilation, spatial_rank, static_cast<size_t>(1));
        strides = align_to_spatial_rank(strides, spatial_rank, static_cast<size_t>(1));
        pads_begin = align_to_spatial_rank(pads_begin, spatial_rank, static_cast<std::ptrdiff_t>(0));
        pads_end = align_to_spatial_rank(pads_end, spatial_rank, static_cast<std::ptrdiff_t>(0));
    }

    if (desc->deformable_mode) {
        ov::op::v8::DeformableConvolution op;
        op.set_group(desc->groups);
        op.set_deformable_group(desc->deformable_groups);
        op.set_dilations(dilation);
        op.set_strides(strides);
        op.set_auto_pad(desc->auto_pad);
        input_shapes.insert(std::next(input_shapes.begin()), impl_param.get_input_layout(1).get_partial_shape());
        output_shapes = ov::op::v8::shape_infer(&op, input_shapes, pads_begin, pads_end);
    } else if (desc->grouped_weights_shape || desc->groups > 1) {
        ov::op::v1::GroupConvolution op;
        op.set_dilations(dilation);
        op.set_strides(strides);
        op.set_auto_pad(desc->auto_pad);
        auto& weights_shape = input_shapes[1];
        // WA for legacy flow, mostly for unit tests as sometimes grouped conv has non-grouped weights
        if (legacy_flow && input_shapes[1].size() == 4 && input_shapes[0].size() == 4) {
            weights_shape.insert(weights_shape.begin(), desc->groups);
            weights_shape[1] /= desc->groups;
        }
        output_shapes = ov::op::v1::shape_infer(&op, input_shapes, pads_begin, pads_end);
    } else {
        ov::op::v1::Convolution op;
        op.set_dilations(dilation);
        op.set_strides(strides);
        op.set_auto_pad(desc->auto_pad);
        output_shapes = ov::op::v1::shape_infer(&op, input_shapes, pads_begin, pads_end);
    }

    return {layout{output_shapes[0], output_type, output_format}};
}

}  // namespace

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(convolution)

layout convolution_inst::calc_output_layout(convolution_node const& node, kernel_impl_params const& impl_param) {
    return calc_output_layout_impl(node, impl_param, true)[0];
}

template<typename ShapeType>
std::vector<layout> convolution_inst::calc_output_layouts(convolution_node const& node, kernel_impl_params const& impl_param) {
    return calc_output_layout_impl(node, impl_param, false);
}

template std::vector<layout> convolution_inst::calc_output_layouts<ov::PartialShape>(convolution_node const& node, const kernel_impl_params& impl_param);

std::string convolution_inst::to_string(convolution_node const& node) {
    auto desc = node.get_primitive();
    auto strd = desc->stride;
    auto groups = node.get_groups();
    auto dilation = desc->dilation;
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;

    std::string w_zp = desc->weights_zero_points.empty() ? "false" : "true";
    std::string a_zp = desc->activations_zero_points.empty() ? "false" : "true";

    json_composite conv_info;
    conv_info.add("stride", cldnn::to_string(strd));
    conv_info.add("padding above", cldnn::to_string(desc->padding_begin));
    conv_info.add("padding below", cldnn::to_string(desc->padding_end));
    conv_info.add("auto pad", cldnn::to_string(desc->auto_pad));
    conv_info.add("groups", groups);
    conv_info.add("dilation", cldnn::to_string(dilation));
    conv_info.add("deformable_groups", desc->deformable_groups);
    conv_info.add("groups", desc->groups);
    conv_info.add("has zero points for weights: ", w_zp);
    conv_info.add("has zero points for activations: ", a_zp);
    node_info->add("convolution info", conv_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

convolution_inst::typed_primitive_inst(network& network, convolution_node const& node) :
    parent(network, node),
    _deform_conv_dep_offset(node.get_deform_conv_dep_offset()) {
    if (node.is_dynamic())
        return;
    OPENVINO_ASSERT(all_not_zeroes(argument->stride), "[GPU] Convolution strides must be positive numbers");
    OPENVINO_ASSERT(all_not_zeroes(argument->dilation), "[GPU] Convolution dilations must be positive numbers");

    auto input_layout = node.get_input_layout();
    auto output_layout = node.get_output_layout();
    auto output_size = output_layout.get_tensor();

    CLDNN_ERROR_NOT_EQUAL(node.id(),
                          "Input number of dimensions",
                          input_layout.get_rank(),
                          "output number of dimensions",
                          output_layout.get_rank(),
                          "Input/output rank mismatch");

    auto filter_inst = node.weights().get_output_layout().convert_to_weights_layout(argument->grouped_weights_shape);

    if (bias_term()) {
        auto bias_inst = node.bias().get_output_layout();
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
                                output_size.feature[0],
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
}  // namespace cldnn
