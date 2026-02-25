//*****************************************************************************
// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
//*****************************************************************************

#include <string>

#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
// helper func - get pad_begin and pad_end
static void get_paddings(const NodeContext& node,
                         ov::Shape& pad_begin,
                         ov::Shape& pad_end,
                         ov::op::PadType& auto_pad,
                         std::string& data_format) {
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
    [pad_depth, pad_height, pad_width] or [pad_depth_front, pad_depth_back,
    pad_height_top, pad_height_bottom, pad_width_left, pad_width_right],
    and when data_format is “NCDHW”, pool_padding can
    be in the form [[0,0], [0,0], [pad_depth_front, pad_depth_back], [pad_height_top,
    pad_height_bottom], [pad_width_left, pad_width_right]]. when
    data_format is “NDHWC”, pool_padding can be in the form
    [[0,0], [pad_depth_front, pad_depth_back], [pad_height_top,
    pad_height_bottom], [pad_width_left, pad_width_right], [0,0]].
    Otherwise, the pool padding size will be a square of an int.*/
    auto paddings = node.get_attribute<std::vector<int32_t>>("paddings");

    switch (paddings.size()) {
    case 3:
        pad_begin =
            Shape{static_cast<size_t>(paddings[0]), static_cast<size_t>(paddings[1]), static_cast<size_t>(paddings[2])};
        pad_end = pad_begin;
        break;
    case 6:
        pad_begin =
            Shape{static_cast<size_t>(paddings[0]), static_cast<size_t>(paddings[2]), static_cast<size_t>(paddings[4])};
        pad_end = Shape{
            static_cast<size_t>(paddings[1]),
            static_cast<size_t>(paddings[3]),
            static_cast<size_t>(paddings[5]),
        };
        break;
    default:
        throw std::runtime_error("Unsupported pooling paddings " + std::to_string(paddings.size()));
    }
}

NamedOutputs pool3d(const NodeContext& node) {
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

    PADDLE_OP_CHECK(node, (pooling_type == "max") || (pooling_type == "avg"), "pool3d: not supported pooling type !");
    PADDLE_OP_CHECK(node, kernel_shape.size() == 1 || kernel_shape.size() == 3, "pool3d: ksize must be 1 or 3!");

    PartialShape input_shape = data.get_partial_shape();

    int32_t input_rank = static_cast<int32_t>(input_shape.rank().get_length());
    PADDLE_OP_CHECK(node, input_rank >= 2, "input tensor rank must be greater than 2");

    auto auto_pad = ov::op::PadType::EXPLICIT;
    ov::Shape pad_begin, pad_end;
    std::string data_format = node.get_attribute<std::string>("data_format", "NCDHW");

    get_paddings(node, pad_begin, pad_end, auto_pad, data_format);

    if (data_format == "NDHWC") {
        data = std::make_shared<default_opset::Transpose>(
            data,
            std::make_shared<default_opset::Constant>(ov::element::i64, Shape{5}, std::vector<int64_t>{0, 4, 1, 2, 3}));
        input_shape = data.get_partial_shape();
    }

    std::vector<Output<Node>> pool_outputs;
    if (global_pooling || (adaptive && std::any_of(kernel_shape.begin(), kernel_shape.end(), [](int32_t i) {
                               return i == 1;
                           }))) {
        if (pooling_type == "max") {
            auto axes = default_opset::Constant::create(ov::element::i64,
                                                        {3},
                                                        {input_rank - 3, input_rank - 2, input_rank - 1});
            pool_outputs = std::make_shared<default_opset::ReduceMax>(data, axes, true)->outputs();
        } else {
            auto axes = default_opset::Constant::create(ov::element::i64,
                                                        {3},
                                                        {input_rank - 3, input_rank - 2, input_rank - 1});
            pool_outputs = std::make_shared<default_opset::ReduceMean>(data, axes, true)->outputs();
        }
    } else if (adaptive) {
        auto pool_size = std::vector<int64_t>(3, 0);

        if (kernel_shape.size() == 1) {
            // Not tested: implemented according to spec, but can't generate real
            // model to test
            pool_size[0] = pool_size[1] = pool_size[2] = kernel_shape[0];
        } else {
            pool_size[0] = kernel_shape[0];
            pool_size[1] = kernel_shape[1];
            pool_size[2] = kernel_shape[2];
        }

        const Output<ov::Node> output_shape =
            default_opset::Constant::create(ov::element::i64, {pool_size.size()}, pool_size);

        if (pooling_type == "max") {
            pool_outputs =
                std::make_shared<default_opset::AdaptiveMaxPool>(data, output_shape, ov::element::i32)->outputs();
        } else {
            pool_outputs = std::make_shared<default_opset::AdaptiveAvgPool>(data, output_shape)->outputs();
        }
    } else {
        auto strides = node.get_attribute<std::vector<int32_t>>("strides");

        size_t kernel_d, kernel_h, kernel_w;
        if (kernel_shape.size() == 1) {
            // Not tested: implemented according to spec, but can't generate real
            // model to test
            kernel_d = kernel_h = kernel_w = kernel_shape[0];
        } else {
            kernel_d = kernel_shape[0];
            kernel_h = kernel_shape[1];
            kernel_w = kernel_shape[2];
        }

        PADDLE_OP_CHECK(node,
                        kernel_d > 0 && kernel_h > 0 && kernel_w > 0,
                        "pool3d kernel shape must be greater than 0");

        // Note: this shape check is only valid when the spatial dim of input_shape
        // is static.
        if (input_shape[2].is_static() && input_shape[3].is_static() && input_shape[4].is_static()) {
            uint64_t input_d = input_shape[input_rank - 3].get_length();
            uint64_t input_h = input_shape[input_rank - 2].get_length();
            uint64_t input_w = input_shape[input_rank - 1].get_length();
            if ((input_d > 0) && (input_d + pad_begin[0] + pad_end[0] < kernel_d)) {
                kernel_d = input_d + pad_begin[0] + pad_end[0];
            }
            if ((input_h > 0) && (input_h + pad_begin[1] + pad_end[1] < kernel_h)) {
                kernel_h = input_h + pad_begin[1] + pad_end[1];
            }
            if ((input_w > 0) && (input_w + pad_begin[2] + pad_end[2] < kernel_w)) {
                kernel_w = input_w + pad_begin[2] + pad_end[2];
            }
        }

        if (pooling_type == "max") {
            pool_outputs = std::make_shared<default_opset::MaxPool>(data,
                                                                    ov::Strides(strides.begin(), strides.end()),
                                                                    ov::Strides{1, 1, 1},
                                                                    pad_begin,
                                                                    pad_end,
                                                                    ov::Shape{kernel_d, kernel_h, kernel_w},
                                                                    rounding_type,
                                                                    auto_pad,
                                                                    ov::element::i32,
                                                                    2)
                               ->outputs();
        } else {
            bool exclude_pad = node.get_attribute<bool>("exclusive", false);
            pool_outputs = std::make_shared<default_opset::AvgPool>(data,
                                                                    ov::Strides(strides.begin(), strides.end()),
                                                                    pad_begin,
                                                                    pad_end,
                                                                    ov::Shape{kernel_d, kernel_h, kernel_w},
                                                                    exclude_pad,
                                                                    rounding_type,
                                                                    auto_pad)
                               ->outputs();
        }
    }

    if (data_format == "NDHWC") {
        pool_outputs[0] = std::make_shared<default_opset::Transpose>(
            pool_outputs[0],
            std::make_shared<default_opset::Constant>(ov::element::i64, Shape{5}, std::vector<int64_t>{0, 2, 3, 4, 1}));
    }

    return NamedOutputs{{"Out", {pool_outputs[0]}}};
}

NamedOutputs pool3d_with_index(const NodeContext& node) {
    auto data = node.get_input("X");
    auto pooling_type = node.get_attribute<std::string>("pooling_type", {});
    auto adaptive = node.get_attribute<bool>("adaptive");
    auto kernel_shape = node.get_attribute<std::vector<int32_t>>("ksize");

    auto rounding_type =
        node.get_attribute<bool>("ceil_mode", false) ? ov::op::RoundingType::CEIL : ov::op::RoundingType::FLOOR;

    if (pooling_type.empty()) {
        pooling_type = "max";
    }

    PADDLE_OP_CHECK(node, (pooling_type == "max") || (pooling_type == "avg"), "pool3d: not supported pooling type !");
    PADDLE_OP_CHECK(node, kernel_shape.size() == 1 || kernel_shape.size() == 3, "pool3d: ksize must be 1 or 3!");

    PartialShape input_shape = data.get_partial_shape();

    int32_t input_rank = static_cast<int32_t>(input_shape.rank().get_length());
    PADDLE_OP_CHECK(node, input_rank >= 2, "input tensor rank must be greater than 2");

    auto auto_pad = ov::op::PadType::EXPLICIT;
    ov::Shape pad_begin, pad_end;
    std::string data_format = node.get_attribute<std::string>("data_format", "NCDHW");

    get_paddings(node, pad_begin, pad_end, auto_pad, data_format);

    if (data_format == "NDHWC") {
        data = std::make_shared<default_opset::Transpose>(
            data,
            std::make_shared<default_opset::Constant>(ov::element::i64, Shape{5}, std::vector<int64_t>{0, 4, 1, 2, 3}));
        input_shape = data.get_partial_shape();
    }

    std::vector<Output<Node>> pool_outputs;
    if (adaptive) {
        auto pool_size = std::vector<int64_t>(3, 0);

        if (kernel_shape.size() == 1) {
            // Not tested: implemented according to spec, but can't generate real
            // model to test
            pool_size[0] = pool_size[1] = pool_size[2] = kernel_shape[0];
        } else {
            pool_size[0] = kernel_shape[0];
            pool_size[1] = kernel_shape[1];
            pool_size[2] = kernel_shape[2];
        }

        const Output<ov::Node> output_shape =
            default_opset::Constant::create(ov::element::i64, {pool_size.size()}, pool_size);

        pool_outputs =
            std::make_shared<default_opset::AdaptiveMaxPool>(data, output_shape, ov::element::i32)->outputs();

    } else {
        auto strides = node.get_attribute<std::vector<int32_t>>("strides");

        size_t kernel_d, kernel_h, kernel_w;
        if (kernel_shape.size() == 1) {
            // Not tested: implemented according to spec, but can't generate real
            // model to test
            kernel_d = kernel_h = kernel_w = kernel_shape[0];
        } else {
            kernel_d = kernel_shape[0];
            kernel_h = kernel_shape[1];
            kernel_w = kernel_shape[2];
        }

        PADDLE_OP_CHECK(node,
                        kernel_d > 0 && kernel_h > 0 && kernel_w > 0,
                        "pool3d kernel shape must be greater than 0");

        // Note: this shape check is only valid when the spatial dim of input_shape
        // is static.
        if (input_shape[2].is_static() && input_shape[3].is_static() && input_shape[4].is_static()) {
            uint64_t input_d = input_shape[input_rank - 3].get_length();
            uint64_t input_h = input_shape[input_rank - 2].get_length();
            uint64_t input_w = input_shape[input_rank - 1].get_length();
            if ((input_d > 0) && (input_d + pad_begin[0] + pad_end[0] < kernel_d)) {
                kernel_d = input_d + pad_begin[0] + pad_end[0];
            }
            if ((input_h > 0) && (input_h + pad_begin[1] + pad_end[1] < kernel_h)) {
                kernel_h = input_h + pad_begin[1] + pad_end[1];
            }
            if ((input_w > 0) && (input_w + pad_begin[2] + pad_end[2] < kernel_w)) {
                kernel_w = input_w + pad_begin[2] + pad_end[2];
            }
        }

        pool_outputs = std::make_shared<default_opset::MaxPool>(data,
                                                                ov::Strides(strides.begin(), strides.end()),
                                                                ov::Strides{1, 1, 1},
                                                                pad_begin,
                                                                pad_end,
                                                                ov::Shape{kernel_d, kernel_h, kernel_w},
                                                                rounding_type,
                                                                auto_pad,
                                                                ov::element::i32,
                                                                2)
                           ->outputs();
    }

    if (data_format == "NDHWC") {
        pool_outputs[0] = std::make_shared<default_opset::Transpose>(
            pool_outputs[0],
            std::make_shared<default_opset::Constant>(ov::element::i64, Shape{5}, std::vector<int64_t>{0, 2, 3, 4, 1}));
    }

    auto output_name = node.get_output_names();
    return NamedOutputs{{"Out", {pool_outputs[0]}}, {"Mask", {pool_outputs[1]}}};
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
