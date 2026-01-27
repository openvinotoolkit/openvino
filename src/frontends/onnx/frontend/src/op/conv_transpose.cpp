// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/core/coordinate_diff.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/subtract.hpp"
#include "utils/convpool.hpp"
#include "utils/reshape.hpp"
using namespace ov::op;
using ov::CoordinateDiff;
using ov::Shape;
using ov::Strides;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
namespace {
ov::Output<ov::Node> make_group_conv_backprop(const ov::Output<ov::Node>& data,
                                              const ov::Output<ov::Node>& filters,
                                              const ov::Strides& strides,
                                              const ov::Strides& dilations,
                                              const ov::CoordinateDiff& pads_begin,
                                              const ov::CoordinateDiff& pads_end,
                                              const ov::op::PadType& auto_pad_type,
                                              const std::vector<std::int64_t>& output_shape,
                                              const std::vector<std::int64_t>& output_padding) {
    if (output_shape.empty()) {
        return std::make_shared<v1::GroupConvolutionBackpropData>(
            data,
            filters,
            strides,
            pads_begin,
            pads_end,
            dilations,
            auto_pad_type,
            ov::CoordinateDiff(std::begin(output_padding), std::end(output_padding)));
    } else {
        return std::make_shared<v1::GroupConvolutionBackpropData>(
            data,
            filters,
            v0::Constant::create(ov::element::i64, ov::Shape{output_shape.size()}, output_shape),
            strides,
            dilations,
            auto_pad_type,
            ov::CoordinateDiff(std::begin(output_padding), std::end(output_padding)));
    }
}

ov::Output<ov::Node> make_conv_backprop(const ov::Output<ov::Node>& data,
                                        const ov::Output<ov::Node>& filters,
                                        const ov::Strides& strides,
                                        const ov::Strides& dilations,
                                        const ov::CoordinateDiff& pads_begin,
                                        const ov::CoordinateDiff& pads_end,
                                        const ov::op::PadType& auto_pad_type,
                                        const std::vector<std::int64_t>& output_shape,
                                        const std::vector<std::int64_t>& output_padding) {
    if (output_shape.empty()) {
        return std::make_shared<v1::ConvolutionBackpropData>(
            data,
            filters,
            strides,
            pads_begin,
            pads_end,
            dilations,
            auto_pad_type,
            ov::CoordinateDiff(std::begin(output_padding), std::end(output_padding)));
    } else {
        return std::make_shared<v1::ConvolutionBackpropData>(
            data,
            filters,
            v0::Constant::create(ov::element::i64, ov::Shape{output_shape.size()}, output_shape),
            strides,
            pads_begin,
            pads_end,
            dilations,
            auto_pad_type,
            ov::CoordinateDiff(std::begin(output_padding), std::end(output_padding)));
    }
}

ov::Output<ov::Node> get_prepared_bias(const ov::Output<ov::Node>& bias, const ov::Output<ov::Node>& conv) {
    // Prepare bias shape [1, C, 1, 1]
    const auto& conv_pshape = conv.get_partial_shape();
    std::shared_ptr<ov::Node> bias_shape_node;

    if (conv_pshape.rank().is_static() && conv_pshape[1].is_static()) {
        ov::Shape new_bias_shape(conv_pshape.rank().get_length(), 1);
        new_bias_shape[1] = conv_pshape[1].get_length();

        bias_shape_node = v0::Constant::create(ov::element::i64, ov::Shape{new_bias_shape.size()}, new_bias_shape);
    } else {
        const auto conv_shape = std::make_shared<v3::ShapeOf>(conv);
        const auto conv_rank = std::make_shared<v3::ShapeOf>(conv_shape);

        // Prepare new bias shape base: [1, 1, 1, 1, ... ]
        const auto one_node = v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
        const auto two_node = v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});
        const auto remaining_shape_length = std::make_shared<v1::Subtract>(conv_rank, two_node);
        const auto remaining_bias_shape_ones = std::make_shared<v3::Broadcast>(one_node, remaining_shape_length);

        const auto C_dim = std::make_shared<v1::StridedSlice>(conv_shape,
                                                              one_node,                  // begin
                                                              two_node,                  // end
                                                              std::vector<int64_t>{0},   // begin mask
                                                              std::vector<int64_t>{0});  // end mask

        // Construct new bias shape: [1, C, 1, 1, ... ]
        bias_shape_node = std::make_shared<v0::Concat>(ov::OutputVector{one_node, C_dim, remaining_bias_shape_ones}, 0);
    }

    return std::make_shared<v1::Reshape>(bias, bias_shape_node, false);
}
}  // namespace

ov::OutputVector conv_transpose(const ov::frontend::onnx::Node& node) {
    const ov::OutputVector& inputs = node.get_ov_inputs();

    CHECK_VALID_NODE(node,
                     inputs.size() == 2 || inputs.size() == 3,
                     "Provided number of inputs is incorrect. The ConvTranspose "
                     "operator expects 2 or 3 inputs.");

    auto data = inputs[0];
    auto filters = inputs[1];

    const auto& data_pshape = data.get_partial_shape();
    const auto& filters_pshape = filters.get_partial_shape();

    std::size_t num_spatial_dims = 0;
    ov::Strides strides, dilations;
    std::pair<ov::CoordinateDiff, ov::CoordinateDiff> paddings;
    ov::op::PadType auto_pad_type = convpool::get_auto_pad(node);

    // Get attirbutes or infer them from input data rank it it's static.
    if (data_pshape.rank().is_static()) {
        num_spatial_dims = data_pshape.rank().get_length() - 2;
    } else if (filters_pshape.rank().is_static()) {
        num_spatial_dims = filters_pshape.rank().get_length() - 2;
    }
    // Otherwise read "kernel_shape" attribute
    else {
        CHECK_VALID_NODE(node,
                         node.has_attribute("kernel_shape"),
                         "\"kernel_shape\" attribute is required if data and "
                         "filter inputs' ranks are dynamic.");
        std::vector<std::size_t> kernel_shape = node.get_attribute_value<std::vector<std::size_t>>("kernel_shape");

        num_spatial_dims = kernel_shape.size();
    }

    strides = convpool::get_strides(node, num_spatial_dims);
    dilations = convpool::get_dilations(node, num_spatial_dims);
    paddings = convpool::get_pads(node, num_spatial_dims);
    ov::CoordinateDiff pads_begin = paddings.first;
    ov::CoordinateDiff pads_end = paddings.second;

    std::vector<std::int64_t> output_shape{node.get_attribute_value<std::vector<std::int64_t>>("output_shape", {})};

    // ONNX spec states that output_shape should contain only spatial dimensions.
    // However, some models incorrectly include batch and channel dimensions.
    // To maintain compatibility with other runtimes (e.g., ONNXRuntime),
    // extract only the spatial dimensions if the shape includes batch and channels.
    if (!output_shape.empty() && output_shape.size() > num_spatial_dims) {
        CHECK_VALID_NODE(node,
                         output_shape.size() == num_spatial_dims + 2,
                         "ConvTranspose output_shape attribute has unexpected number of dimensions. "
                         "Expected ",
                         num_spatial_dims,
                         " spatial dimensions or ",
                         num_spatial_dims + 2,
                         " dimensions (with batch and channels), but got ",
                         output_shape.size());

        // Extract last num_spatial_dims elements (the spatial dimensions)
        // For example: [1, 16, 32, 32] -> [32, 32] for 2D convolution
        output_shape = std::vector<std::int64_t>(output_shape.end() - num_spatial_dims, output_shape.end());
    }

    std::vector<std::int64_t> output_padding{
        node.get_attribute_value<std::vector<std::int64_t>>("output_padding",
                                                            std::vector<std::int64_t>(num_spatial_dims, 0))};

    // ONNX Runtime behavior: When output_shape is explicitly provided, recalculate padding
    // to achieve the desired output shape. This is necessary because ONNX models may specify
    // output_shape without corresponding padding values.
    // Note: When both output_shape and auto_pad are specified, output_shape takes precedence
    // and auto_pad is effectively ignored (padding is computed to match output_shape).
    // This matches ONNX Runtime behavior.
    if (!output_shape.empty() && data_pshape.is_static() && filters_pshape.is_static()) {
        const ov::Shape data_static_shape = data_pshape.to_shape();
        const ov::Shape filters_static_shape = filters_pshape.to_shape();

        // Recalculate padding for each spatial dimension to achieve output_shape
        for (size_t i = 0; i < num_spatial_dims; ++i) {
            const int64_t in_size = data_static_shape[2 + i];
            const int64_t kernel_size = filters_static_shape[2 + i];
            const int64_t stride = strides[i];
            const int64_t dilation = dilations[i];
            const int64_t adj = output_padding[i];
            const int64_t out_size = output_shape[i];

            // Formula from ONNX Runtime: ComputeTotalPad
            // total_pad = max(0, (in_size - 1) * stride + adj + (kernel - 1) * dilation + 1 - out_size)
            const int64_t total_pad =
                std::max<int64_t>(0, (in_size - 1) * stride + adj + (kernel_size - 1) * dilation + 1 - out_size);

            // Distribute padding: for NOTSET/EXPLICIT, put more on head when odd
            // (ONNX Runtime default for non-SAME modes)
            pads_begin[i] = total_pad - total_pad / 2;
            pads_end[i] = total_pad / 2;
        }

        // Override auto_pad_type to EXPLICIT since we've calculated explicit pads
        auto_pad_type = ov::op::PadType::EXPLICIT;
    }
    // Calculate padding for auto_pad modes when output_shape is not provided
    // Per ONNX spec, for ConvTranspose with SAME_UPPER/SAME_LOWER: output = input * stride
    else if (output_shape.empty() &&
             (auto_pad_type == ov::op::PadType::SAME_UPPER || auto_pad_type == ov::op::PadType::SAME_LOWER)) {
        // Only calculate if shapes are static (required for padding calculation)
        if (data_pshape.is_static() && filters_pshape.is_static()) {
            const ov::Shape data_static_shape = data_pshape.to_shape();
            const ov::Shape filters_static_shape = filters_pshape.to_shape();

            // Calculate padding to satisfy ONNX spec: output = input * stride
            convpool::calculate_transpose_auto_pads(
                data_static_shape,
                filters_static_shape,
                strides,
                dilations,
                auto_pad_type,
                ov::CoordinateDiff(std::begin(output_padding), std::end(output_padding)),
                pads_begin,
                pads_end);

            // Override auto_pad_type to EXPLICIT since we've calculated explicit pads
            auto_pad_type = ov::op::PadType::EXPLICIT;
        }
    }

    int64_t groups{node.get_attribute_value<int64_t>("group", 1)};

    CHECK_VALID_NODE(node, groups >= 0, "Incorrect value of 'group' attribute: ", groups);

    ov::Output<ov::Node> conv_node;

    if (groups > 1) {
        filters = convpool::get_reshaped_filters(filters, groups);
        conv_node = make_group_conv_backprop(data,
                                             filters,
                                             strides,
                                             dilations,
                                             pads_begin,
                                             pads_end,
                                             auto_pad_type,
                                             output_shape,
                                             output_padding);
    } else {
        conv_node = make_conv_backprop(data,
                                       filters,
                                       strides,
                                       dilations,
                                       pads_begin,
                                       pads_end,
                                       auto_pad_type,
                                       output_shape,
                                       output_padding);
    }

    // no bias param
    if (inputs.size() < 3) {
        return {conv_node};
    }
    const auto reshaped_bias = get_prepared_bias(inputs[2], conv_node);

    return {std::make_shared<v1::Add>(conv_node, reshaped_bias)};
}
ONNX_OP("ConvTranspose", OPSET_SINCE(1), ai_onnx::opset_1::conv_transpose);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
