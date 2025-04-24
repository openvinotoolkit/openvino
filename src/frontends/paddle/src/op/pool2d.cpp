//*****************************************************************************
// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
//*****************************************************************************

#include "openvino/frontend/paddle/node_context.hpp"
#include "openvino/opsets/opset6.hpp"
#include "openvino/opsets/opset8.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
// helper func - get pad_begin and pad_end
static void get_paddings(const NodeContext& node, ov::Shape& pad_begin, ov::Shape& pad_end, ov::op::PadType& auto_pad) {
    if (node.has_attribute("padding_algorithm")) {
        auto pad_algo = node.get_attribute<std::string>("padding_algorithm");
        if (pad_algo == "SAME") {
            auto_pad = ov::op::PadType::SAME_UPPER;
        } else if (pad_algo == "VALID") {
            auto_pad = ov::op::PadType::VALID;
        } else if (pad_algo == "EXPLICIT") {
            auto_pad = ov::op::PadType::EXPLICIT;
        } else {
            throw std::runtime_error("Unsupported pooling padding_algorithm " + pad_algo);
        }
    } else {
        // adaptive_maxpool with no such attr.
        auto_pad = ov::op::PadType::EXPLICIT;
    }

    /*If pool padding size is a tuple or list, it could be in three forms:
    [pad_height, pad_width] or [pad_height_top, pad_height_bottom, pad_width_left,
    pad_width_right], and when data_format is “NCHW”, pool_padding can be in the
    form [[0,0], [0,0], [pad_height_top, pad_height_bottom], [pad_width_left,
    pad_width_right]]. when data_format is “NHWC”, pool_padding can be in the form
    [[0,0], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right],
    [0,0]]. Otherwise, the pool padding size will be a square of an int.*/
    auto paddings = node.get_attribute<std::vector<int32_t>>("paddings");

    // Default is empty for 'adaptive max pooling'
    auto data_format = node.get_attribute<std::string>("data_format", {});

    // TODO: need to support NHWC input #55483
    switch (paddings.size()) {
    case 2:
        pad_begin = Shape{static_cast<size_t>(paddings[0]), static_cast<size_t>(paddings[1])};
        pad_end = pad_begin;
        break;
    case 4:
        pad_begin = Shape{static_cast<size_t>(paddings[0]), static_cast<size_t>(paddings[2])};
        pad_end = Shape{static_cast<size_t>(paddings[1]), static_cast<size_t>(paddings[3])};
        break;
    default:
        throw std::runtime_error("Unsupported pooling paddings " + std::to_string(paddings.size()));
    }
}

NamedOutputs pool2d(const NodeContext& node) {
    auto data = node.get_input("X");

    auto pooling_type = node.get_attribute<std::string>("pooling_type", {});
    auto global_pooling = node.get_attribute<bool>("global_pooling");
    auto adaptive = node.get_attribute<bool>("adaptive");
    auto kernel_shape = node.get_attribute<std::vector<int32_t>>("ksize");

    auto rounding_type =
        node.get_attribute<bool>("ceil_mode", false) ? ov::op::RoundingType::CEIL : ov::op::RoundingType::FLOOR;

    if (pooling_type.empty()) {
        pooling_type = "max";
    }

    PADDLE_OP_CHECK(node, (pooling_type == "max") || (pooling_type == "avg"), "pool2d: not supported pooling type !");
    PADDLE_OP_CHECK(node, kernel_shape.size() == 1 || kernel_shape.size() == 2, "pool2d: ksize must be 1 or 2!");

    PartialShape input_shape = data.get_partial_shape();

    int32_t input_rank = static_cast<int32_t>(input_shape.rank().get_length());
    PADDLE_OP_CHECK(node, input_rank >= 2, "input tensor rank must be greater than 2");

    auto auto_pad = ov::op::PadType::EXPLICIT;
    ov::Shape pad_begin, pad_end;
    get_paddings(node, pad_begin, pad_end, auto_pad);

    if (global_pooling || (adaptive && std::any_of(kernel_shape.begin(), kernel_shape.end(), [](int32_t i) {
                               return i == 1;
                           }))) {
        if (pooling_type == "max") {
            auto axes = ov::opset6::Constant::create(ov::element::i64, {2}, {input_rank - 2, input_rank - 1});
            return node.default_single_output_mapping({std::make_shared<ov::opset6::ReduceMax>(data, axes, true)},
                                                      {"Out"});
        } else {
            auto axes = ov::opset6::Constant::create(ov::element::i64, {2}, {input_rank - 2, input_rank - 1});
            return node.default_single_output_mapping({std::make_shared<ov::opset6::ReduceMean>(data, axes, true)},
                                                      {"Out"});
        }
    } else if (adaptive) {
        auto pool_size = std::vector<int64_t>(2, 0);

        if (kernel_shape.size() == 1) {
            // Not tested: implemented according to spec, but can't generate real
            // model to test
            pool_size[0] = pool_size[1] = kernel_shape[0];
        } else {
            pool_size[0] = kernel_shape[0];
            pool_size[1] = kernel_shape[1];
        }

        const Output<ov::Node> output_shape =
            ov::opset6::Constant::create(ov::element::i64, {pool_size.size()}, pool_size);

        if (pooling_type == "max") {
            std::vector<Output<Node>> pool_outputs;
            pool_outputs =
                std::make_shared<ov::opset8::AdaptiveMaxPool>(data, output_shape, ov::element::i32)->outputs();
            NamedOutputs outputs;
            outputs["Out"] = {pool_outputs[0]};
            outputs["Mask"] = {pool_outputs[1]};
            return outputs;
        } else {
            return node.default_single_output_mapping(
                {std::make_shared<ov::opset8::AdaptiveAvgPool>(data, output_shape)},
                {"Out"});
        }
    } else {
        auto strides = node.get_attribute<std::vector<int32_t>>("strides");
        auto paddings = node.get_attribute<std::vector<int32_t>>("paddings");

        size_t kernel_h, kernel_w;
        if (kernel_shape.size() == 1) {
            // Not tested: implemented according to spec, but can't generate real
            // model to test
            kernel_h = kernel_w = kernel_shape[0];
        } else {
            kernel_h = kernel_shape[0];
            kernel_w = kernel_shape[1];
        }

        PADDLE_OP_CHECK(node, kernel_h > 0 && kernel_w > 0, "pool2d kernel shape must be greater than 0");

        // Note: this shape check is only valid when the spatial dim of input_shape
        // is static.
        if (input_shape[2].is_static() && input_shape[3].is_static()) {
            uint64_t input_h = input_shape[input_rank - 2].get_length();
            uint64_t input_w = input_shape[input_rank - 1].get_length();
            if ((input_h > 0) && (input_h + pad_begin[0] + pad_end[0] < kernel_h)) {
                kernel_h = input_h + pad_begin[0] + pad_end[0];
            }
            if ((input_w > 0) && (input_w + pad_begin[1] + pad_end[1] < kernel_w)) {
                kernel_w = input_w + pad_begin[1] + pad_end[1];
            }
        }

        if (pooling_type == "max") {
            return node.default_single_output_mapping(
                {std::make_shared<ov::opset6::MaxPool>(data,
                                                       ov::Strides(strides.begin(), strides.end()),
                                                       pad_begin,
                                                       pad_end,
                                                       ov::Shape{kernel_h, kernel_w},
                                                       rounding_type,
                                                       auto_pad)},
                {"Out"});
        } else {
            bool exclude_pad = node.get_attribute<bool>("exclusive", false);
            return node.default_single_output_mapping(
                {std::make_shared<ov::opset6::AvgPool>(data,
                                                       ov::Strides(strides.begin(), strides.end()),
                                                       pad_begin,
                                                       pad_end,
                                                       ov::Shape{kernel_h, kernel_w},
                                                       exclude_pad,
                                                       rounding_type,
                                                       auto_pad)},
                {"Out"});
        }
    }
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
