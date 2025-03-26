// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <string>

#include "convolution_backprop_shape_inference.hpp"
#include "deconvolution_inst.h"
#include "group_convolution_backprop_shape_inference.hpp"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include "primitive_type_base.h"
#include "sliding_window_utils.hpp"

using namespace ov::intel_gpu;

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(deconvolution)

layout deconvolution_inst::calc_output_layout(deconvolution_node const& node, kernel_impl_params const& impl_param) {
    assert(static_cast<bool>(impl_param.desc->output_data_types[0]) == false &&
           "Output data type forcing is not supported for deconvolution_node!");
    auto desc = impl_param.typed_desc<deconvolution>();

    auto input_layout = impl_param.get_input_layout();
    auto weights_layout = *impl_param.weights_layout;
    weights_layout = weights_layout.convert_to_weights_layout(desc->grouped_weights_shape);

    auto data_type = input_layout.data_type;
    if ((input_layout.data_type == data_types::i8 || input_layout.data_type == data_types::u8) && !impl_param.has_fused_primitives()) {
        data_type = data_types::f32;
    }

    if (impl_param.has_fused_primitives()) {
        data_type = impl_param.get_output_element_type();
    }

    auto pad = desc->pad;
    auto strd = desc->stride;

    int32_t number_of_features = weights_layout.group() * weights_layout.ofm();

    format out_fmt = input_layout.format;
    if (node.get_preferred_impl_type() == impl_types::onednn && node.get_preferred_output_fmt() != format::any) {
        out_fmt = node.get_preferred_output_fmt();
    }

    if (desc->with_output_size) {
        CLDNN_ERROR_LESS_OR_EQUAL_THAN(desc->id,
                                       "User-defined output spatial X",
                                       desc->output_size.spatial[0],
                                       "value 0",
                                       0,
                                       "User-defined size of output layout must be positive (>= 1)");
        CLDNN_ERROR_LESS_OR_EQUAL_THAN(desc->id,
                                       "User-defined output spatial Y",
                                       desc->output_size.spatial[1],
                                       "value 0",
                                       0,
                                       "User-defined size of output layout must be positive (>= 1)");
        CLDNN_ERROR_LESS_OR_EQUAL_THAN(desc->id,
                                       "User-defined output spatial Z",
                                       desc->output_size.spatial[2],
                                       "value 0",
                                       0,
                                       "User-defined size of output layout must be positive (>= 1)");

        tensor output_size(input_layout.batch(),
                           number_of_features,
                           desc->output_size.spatial[0],
                           desc->output_size.spatial[1],
                           desc->output_size.spatial[2]);
        return {data_type, out_fmt, output_size};
    }

    int32_t off_factor = -2;
    size_t spatial_dims = input_layout.get_spatial_rank();
    CLDNN_ERROR_GREATER_THAN(desc->id,
                             "number of spatial dimensions",
                             spatial_dims,
                             "expected number of dimensions",
                             3,
                             "As for now, deconvolutions with more than 3 dimensions are not supported");

    int32_t x = static_cast<int32_t>(
        off_factor * pad[pad.size() - 1] + (input_layout.spatial(0) - 1) * strd[strd.size() - 1] + weights_layout.spatial(0));
    int32_t y = 1;
    if (spatial_dims > 1) {
        y = static_cast<int32_t>(
            off_factor * pad[pad.size() - 2] + (input_layout.spatial(1) - 1) * strd[strd.size() - 2] + weights_layout.spatial(1));
    }
    int32_t z = 1;
    if (spatial_dims > 2) {
        z = static_cast<int32_t>(
            off_factor * pad[pad.size() - 3] + (input_layout.spatial(2) - 1) * strd[strd.size() - 3] + weights_layout.spatial(2));
    }

    tensor output_size(input_layout.batch(),
                       number_of_features, x, y, z);
    return {data_type, out_fmt, output_size};
}

template<typename ShapeType>
std::vector<layout> deconvolution_inst::calc_output_layouts(deconvolution_node const& node, const kernel_impl_params& impl_param) {
    auto desc = impl_param.typed_desc<deconvolution>();

    auto input_layout = impl_param.get_input_layout(0);
    auto weights_layout = *impl_param.weights_layout;
    weights_layout = weights_layout.convert_to_weights_layout(desc->grouped_weights_shape);

    if (input_layout.is_dynamic())
        return {layout{ShapeType::dynamic(input_layout.get<ShapeType>().rank()), input_layout.data_type, input_layout.format}};

    auto input_type = input_layout.data_type;
    auto output_type = input_type;
    if ((input_type == data_types::i8 || input_type == data_types::u8) && !impl_param.has_fused_primitives()) {
        output_type = data_types::f32;
    }

    if (impl_param.has_fused_primitives()) {
        output_type = impl_param.get_output_element_type();
    }

    auto strides = desc->stride;
    auto dilations = desc->dilations;
    auto pads_begin = desc->pads_begin;
    auto pads_end = desc->pads_end;
    auto output_padding = desc->out_padding;
    auto output_partial_shape = desc->output_partial_shape;

    int32_t number_of_features = weights_layout.group() * weights_layout.ofm();

    format out_fmt = input_layout.format;
    if (node.get_preferred_impl_type() == impl_types::onednn && node.get_preferred_output_fmt() != format::any) {
        out_fmt = node.get_preferred_output_fmt();
    }

    if (!node.get_program().is_new_shape_infer() && desc->with_output_size) {
        CLDNN_ERROR_LESS_OR_EQUAL_THAN(desc->id,
                                       "User-defined output spatial X",
                                       desc->output_size.spatial[0],
                                       "value 0",
                                       0,
                                       "User-defined size of output layout must be positive (>= 1)");
        CLDNN_ERROR_LESS_OR_EQUAL_THAN(desc->id,
                                       "User-defined output spatial Y",
                                       desc->output_size.spatial[1],
                                       "value 0",
                                       0,
                                       "User-defined size of output layout must be positive (>= 1)");
        CLDNN_ERROR_LESS_OR_EQUAL_THAN(desc->id,
                                       "User-defined output spatial Z",
                                       desc->output_size.spatial[2],
                                       "value 0",
                                       0,
                                       "User-defined size of output layout must be positive (>= 1)");

        tensor output_size(input_layout.batch(),
                           number_of_features,
                           desc->output_size.spatial[0],
                           desc->output_size.spatial[1],
                           desc->output_size.spatial[2]);
        return {layout{output_type, out_fmt, output_size}};
    }

    std::vector<ShapeType> input_shapes = {
        input_layout.get<ShapeType>()
    };
    std::vector<ShapeType> output_shapes;
    auto& memory_deps = impl_param.memory_deps;
    // Dimensions order of weights is IOYX, but the selected format is OIYX by default and I/O dimensions are
    // already swapped when creating constant op. So we need to swap I/O dimensions according to the original
    // dimension order for shape inference.
    auto weights_pshape = weights_layout.get_partial_shape();
    if (desc->groups > 1) {
        ov::op::v1::GroupConvolutionBackpropData op;
        op.set_strides(strides);
        op.set_dilations(dilations);
        op.set_output_padding(output_padding);
        op.set_auto_pad(ov::op::PadType::EXPLICIT);
        std::swap(weights_pshape[2], weights_pshape[1]);
        input_shapes.push_back(weights_pshape);
        if (output_partial_shape.size() != 0) {
            op.set_output_shape(output_partial_shape.to_shape());
            input_shapes.push_back(ov::Shape{output_partial_shape.size()});
            output_shapes = ov::op::v1::shape_infer(&op, input_shapes, pads_begin, pads_end);
        } else if (memory_deps.count(2)) {
            auto mem = memory_deps.at(2);
            auto dims = read_vector<int64_t>(mem, impl_param.get_stream());
            auto dims_shape = ov::Shape{dims.size()};
            auto const_data =
                std::unordered_map<size_t, ov::Tensor>{{2, ov::Tensor(ov::element::i64, dims_shape, dims.data())}};
            input_shapes.push_back(dims_shape);
            output_shapes =
                ov::op::v1::shape_infer(&op, input_shapes, pads_begin, pads_end, ov::make_tensor_accessor(const_data));
        } else {
            output_shapes = ov::op::v1::shape_infer(&op, input_shapes, pads_begin, pads_end);
        }
    } else {
        ov::op::v1::ConvolutionBackpropData op;
        op.set_strides(strides);
        op.set_dilations(dilations);
        op.set_output_padding(output_padding);
        op.set_auto_pad(ov::op::PadType::EXPLICIT);
        std::swap(weights_pshape[1], weights_pshape[0]);
        input_shapes.push_back(weights_pshape);
        if (output_partial_shape.size() != 0) {
            op.set_output_shape(output_partial_shape.to_shape());
            input_shapes.push_back(ov::Shape{output_partial_shape.size()});
            output_shapes = ov::op::v1::shape_infer(&op, input_shapes, pads_begin, pads_end);
        } else if ((desc->output_shape_id != "" || desc->output_partial_shape.size() > 0) && memory_deps.count(2)) {
            auto mem = memory_deps.at(2);
            auto dims = read_vector<int64_t>(mem, impl_param.get_stream());
            auto dims_shape = ov::Shape{dims.size()};
            auto const_data =
                std::unordered_map<size_t, ov::Tensor>{{2, ov::Tensor(ov::element::i64, dims_shape, dims.data())}};
            input_shapes.push_back(dims_shape);

            output_shapes =
                ov::op::v1::shape_infer(&op, input_shapes, pads_begin, pads_end, ov::make_tensor_accessor(const_data));
        } else {
            output_shapes = ov::op::v1::shape_infer(&op, input_shapes, pads_begin, pads_end);
        }
    }
    return {layout{output_shapes[0], output_type, out_fmt.value}};
}

template std::vector<layout> deconvolution_inst::calc_output_layouts<ov::PartialShape>(deconvolution_node const& node,
                                                                                       const kernel_impl_params& impl_param);

std::string deconvolution_inst::to_string(deconvolution_node const& node) {
    auto desc = node.get_primitive();
    auto strd = desc->stride;
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;

    json_composite deconv_info;
    deconv_info.add("stride", cldnn::to_string(strd));
    deconv_info.add("pad", cldnn::to_string(desc->pad));
    deconv_info.add("groups", desc->groups);
    if (desc->with_output_size) {
        json_composite ud_out_size_info;
        ud_out_size_info.add("size", desc->output_size.to_string());
        deconv_info.add("with_user_defined_output_size", ud_out_size_info);
    }
    std::stringstream ss_weights;
    ss_weights << node.weights().id();
    ss_weights << ", count: " << node.weights().get_output_layout().count();
    deconv_info.add("weights", ss_weights.str());
    if (node.bias_term()) {
        std::stringstream ss_biases;
        ss_biases << node.bias().id();
        ss_biases << ", count: " << node.bias().get_output_layout().count();
        deconv_info.add("bias", ss_biases.str());
    }

    node_info->add("deconvolution info", deconv_info);
    node_info->dump(primitive_description);
    return primitive_description.str();
}

deconvolution_inst::typed_primitive_inst(network& network, deconvolution_node const& node)
    : parent(network, node) {
    if (node.is_dynamic())
        return;
    auto stride = argument->stride;
    auto pad = argument->pad;

    auto input_layout = node.get_input_layout();
    auto output_layout = node.get_output_layout();

    CLDNN_ERROR_NOT_EQUAL(node.id(),
                          "Input size",
                          input_layout.get_rank(),
                          "output size",
                          output_layout.get_rank(),
                          "Input/output number of dimension does not match.");
    if (!node.get_program().is_new_shape_infer()) {
        CLDNN_ERROR_NOT_EQUAL(node.id(),
                            "Stride size",
                            stride.size(),
                            "output size",
                            output_layout.get_spatial_rank(),
                            "Stride/output number of dimension does not match.");
        CLDNN_ERROR_NOT_EQUAL(node.id(),
                            "Input offset size",
                            pad.size(),
                            "input number of dimensions",
                            output_layout.get_spatial_rank(),
                            "");
    }

    auto filter_inst = node.weights().get_output_layout().convert_to_weights_layout(argument->grouped_weights_shape);

    if (argument->bias.size() != 0) {
        auto bias_inst = node.bias().get_output_layout();
        CLDNN_ERROR_NOT_EQUAL(node.id(),
                                "Bias batch[0]",
                                bias_inst.batch(),
                                "dimension size",
                                1,
                                "Batch[0] of bias should be 1. Bias isn't 1D vector.");
        CLDNN_ERROR_NOT_EQUAL(node.id(),
                                "Bias feature[0]",
                                bias_inst.feature(),
                                "output feature size",
                                output_layout.feature(),
                                "Biases/output feature maps number does not match.");
        CLDNN_ERROR_NOT_EQUAL(node.id(),
                                "Bias spatial[2]",
                                bias_inst.spatial(2),
                                "dimension size",
                                1,
                                "Spatial[2] of bias should be 1. Bias isn't 1D vector.");
        CLDNN_ERROR_NOT_EQUAL(node.id(),
                                "Bias spatial[1]",
                                bias_inst.spatial(1),
                                "dimension size",
                                1,
                                "Spatial[1] of bias should be 1. Bias isn't 1D vector.");
        CLDNN_ERROR_NOT_EQUAL(node.id(),
                                "Bias spatial[0]",
                                bias_inst.spatial(0),
                                "dimension size",
                                1,
                                "Spatial[0] of bias should be 1. Bias isn't 1D vector.");
    }

    CLDNN_ERROR_NOT_EQUAL(node.id(),
                            "Weights feature maps number",
                            filter_inst.ifm() * filter_inst.group(),
                            "input feature maps number",
                            input_layout.feature(),
                            "Weights/ifm mismatch");
}

}  // namespace cldnn
