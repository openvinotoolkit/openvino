// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/cc/ngraph/itt.hpp>

#include "transformations/convert_padded2valid_conv.hpp"

#include <memory>

#include <ngraph/opsets/opset7.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pattern/op/or.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pass/manager.hpp>
#include <ie_common.h>


using namespace GNAPluginNS;

NGRAPH_RTTI_DEFINITION(ConvertPadded2ValidConv, "ConvertPadded2ValidConv", 0);

struct ConvData {
    size_t input_height;
    size_t input_width;
    size_t input_channel_count;
    size_t filter_count;
    size_t pads_begin_width;
    size_t pads_begin_height;
    size_t pads_end_width;
    size_t pads_end_height;
    ngraph::op::PadType padding_type;
    ngraph::element::Type element_type;
};

static bool VerifyAndGetConvParams(std::shared_ptr<ngraph::opset7::Convolution> conv, ConvData& conv_data) {
    const auto& input = conv->input_value(0);

    // We support only 2D conv batch 1
    if (conv->get_dilations().size() != 2 ||
        conv->get_strides().size() != 2 ||
        input.get_shape()[0] != 1) {
        return false;
    }

    conv_data.padding_type = conv->get_auto_pad();
    conv_data.input_channel_count = conv->input_value(0).get_shape()[1];
    conv_data.input_height = conv->input_value(0).get_shape()[2];
    conv_data.input_width = conv->input_value(0).get_shape()[3];
    conv_data.filter_count = conv->input_value(1).get_shape()[0];
    conv_data.pads_begin_height = conv->get_pads_begin()[0];
    conv_data.pads_begin_width = conv->get_pads_begin()[1];
    conv_data.pads_end_height = conv->get_pads_end()[0];
    conv_data.pads_end_width = conv->get_pads_end()[1];
    conv_data.element_type = conv->get_element_type();

    return conv_data.pads_begin_height || conv_data.pads_end_height || conv_data.pads_begin_width || conv_data.pads_end_width;
}

static bool TransposeOrderMatches(std::shared_ptr<ngraph::opset7::Transpose> transpose, std::vector<size_t> order) {
    if (!transpose)
        return false;
    const ngraph::Output<ngraph::Node>& transpose_order = transpose->input_value(1);
    auto transpose_order_dim = transpose_order.get_shape().size();

    if (transpose_order_dim != 1 || transpose_order.get_shape()[0] != order.size())
        return false;

    auto const_with_order_values = std::dynamic_pointer_cast<ngraph::opset7::Constant>(transpose_order.get_node_shared_ptr());
    if (!const_with_order_values)
        return false;

    const auto data = const_with_order_values->cast_vector<size_t>();
    if (data.empty())
        return false;

    if (!std::equal(order.begin(), order.end(), data.begin()))
        return false;

    return true;
}

static bool VerifyBias(std::shared_ptr<ngraph::opset7::Add> bias, const size_t& filter_count) {
    auto add_const = std::dynamic_pointer_cast<ngraph::opset7::Constant>(bias->input_value(0).get_node_shared_ptr());

    // We need to check both inputs of Add when looking for constant
    if (!add_const)
        add_const = std::dynamic_pointer_cast<ngraph::opset7::Constant>(bias->input_value(1).get_node_shared_ptr());

    // The add may be a normal add not convolution bias, then we just go further
    return (add_const && shape_size(add_const->get_shape()) == filter_count);
}

static std::shared_ptr<ngraph::opset7::StridedSlice> FlatCrop(ngraph::Output<ngraph::Node> input, size_t offset, size_t size) {
    return std::make_shared<ngraph::opset7::StridedSlice>(
        input, // data
        ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {(size_t)0, offset}), // begin sice index
        ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {(size_t)0, offset + size}), // end slice index
        ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {(size_t)1, (size_t)1}), // strides
        std::vector<int64_t>{1, 0},  // begin mask
        std::vector<int64_t>{1, 0}); // end mask
}

static void InsertPadding(ngraph::OutputVector& input_rows_to_concat, size_t size, const std::shared_ptr<ngraph::opset7::Convolution>& conv,
    const std::shared_ptr<ngraph::opset7::Constant> padding_const, size_t biggest_padding) {

    if (size == biggest_padding) {
        input_rows_to_concat.push_back(padding_const);
    } else {
        auto slice = FlatCrop(padding_const, 0, size);
        copy_runtime_info(conv, slice);
        input_rows_to_concat.push_back(slice);
    }
}

static std::shared_ptr<ngraph::Node> CreatePaddedNet(std::shared_ptr<ngraph::opset7::Transpose> leading_transpose,
    std::shared_ptr<ngraph::opset7::Convolution> conv, const ConvData& conv_data) {
    size_t flat_left_padding = conv_data.input_channel_count * conv_data.pads_begin_width;
    size_t flat_right_padding = conv_data.input_channel_count * conv_data.pads_end_width;
    size_t padded_row_size = flat_left_padding + conv_data.input_channel_count * conv_data.input_width + flat_right_padding;
    size_t flat_top_padding = padded_row_size * conv_data.pads_begin_height;
    size_t flat_bottom_padding = padded_row_size * conv_data.pads_end_height;
    size_t biggest_padding = std::max(std::max(flat_left_padding, flat_right_padding), std::max(flat_top_padding, flat_bottom_padding));

    if (conv_data.input_height > 1 && (flat_top_padding > 1 || flat_bottom_padding > 1)) {
        biggest_padding = biggest_padding > padded_row_size ? biggest_padding : padded_row_size;
    }

    auto flat_input = std::make_shared<ngraph::opset7::Reshape>(leading_transpose->input_value(0),
        ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{2},
            ngraph::Shape{1ull, shape_size(leading_transpose->input_value(0).get_shape())}), false);

    // Constant with zero padding
    auto const_holding_padding = std::make_shared<ngraph::opset7::Constant>(conv_data.element_type, ngraph::Shape{1, biggest_padding}, 0);

    copy_runtime_info(conv, const_holding_padding);
    std::shared_ptr<ngraph::Node> original_row = flat_input;
    ngraph::OutputVector input_rows_to_concat;

    // Add top padding
    for (size_t p = 0; p < conv_data.pads_begin_height; p++) {
        InsertPadding(input_rows_to_concat, padded_row_size, conv, const_holding_padding, biggest_padding);
    }

    if (flat_left_padding || flat_right_padding) {
        // Pad every row of input plain if neccessary
        for (size_t h = 0; h < conv_data.input_height; h++) {
            // left padding     input     right padding
            //     |              |           |
            //     +--------------+-----------+
            //                    |
            //                 concat

            if (conv_data.input_height > 1)
                original_row = FlatCrop(flat_input, h * conv_data.input_width * conv_data.input_channel_count,
                    conv_data.input_width * conv_data.input_channel_count);
            copy_runtime_info(conv, original_row);

            ngraph::OutputVector single_row_concat_inputs;
            if (flat_left_padding) {
                InsertPadding(single_row_concat_inputs, flat_left_padding, conv, const_holding_padding, biggest_padding);
            }
            single_row_concat_inputs.push_back(original_row);
            if (flat_right_padding) {
                InsertPadding(single_row_concat_inputs, flat_right_padding, conv, const_holding_padding, biggest_padding);
            }
            auto padded_row_concat = std::make_shared<ngraph::opset7::Concat>(single_row_concat_inputs, 1);
            copy_runtime_info(conv, padded_row_concat);
            input_rows_to_concat.push_back(padded_row_concat);
        }
    } else {
        copy_runtime_info(conv, original_row);
        input_rows_to_concat.push_back(original_row);
    }

    // Bottom padding
    for (size_t p = 0; p < conv_data.pads_end_height; p++) {
        InsertPadding(input_rows_to_concat, padded_row_size, conv, const_holding_padding, biggest_padding);
    }

    auto padded_input_plane = std::make_shared<ngraph::opset7::Concat>(input_rows_to_concat, 1);
    copy_runtime_info(conv, padded_input_plane);
    return padded_input_plane;
}

static void GeneratePadding(std::shared_ptr<ngraph::opset7::Transpose> leading_transpose,
    std::shared_ptr<ngraph::opset7::Convolution> conv, const ConvData& conv_data) {
    // Add padding where neccessary

    // padding
    // padding
    // ... row ...
    // ... row ...
    // ...........
    // ... row ...
    // padding
    // padding
    auto padded_input_plane = CreatePaddedNet(leading_transpose, conv, conv_data);

    auto padded_input_plane_reshaped = std::make_shared<ngraph::opset7::Reshape>(padded_input_plane,
        ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {static_cast<size_t>(1),
            conv_data.pads_begin_height + conv_data.input_height + conv_data.pads_end_height,
            conv_data.pads_begin_width + conv_data.input_width + conv_data.pads_end_width,
            conv_data.input_channel_count}), false);

    // NHWC => NCHW
    auto transposed2chw = std::make_shared<ngraph::opset7::Transpose>(padded_input_plane_reshaped,
        ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0ull, 3ull, 1ull, 2ull})->output(0));

    auto conv_copy = std::make_shared<ngraph::opset7::Convolution>(
        transposed2chw->output(0),
        conv->input_value(1),
        conv->get_strides(),
        ngraph::CoordinateDiff{0, 0},
        ngraph::CoordinateDiff{0, 0},
        conv->get_dilations(),
        ngraph::op::PadType::EXPLICIT);

    replace_node(conv, conv_copy);
}

static bool Convert(std::shared_ptr<ngraph::Node> leading_transpose,
    std::shared_ptr<ngraph::Node> conv,
    std::shared_ptr<ngraph::Node> trailing_transpose,
    std::shared_ptr<ngraph::Node> bias) {

    ConvData conv_data;

    if (!VerifyAndGetConvParams(std::dynamic_pointer_cast<ngraph::opset7::Convolution>(conv), conv_data))
        return false;

    // We are looking for Transpose(NHWC->NCHW) => Conv => Transpose(NCHW->NHWC)
    // or similar cases, so required network must be in NHWC order like in TF
    if (!TransposeOrderMatches(std::dynamic_pointer_cast<ngraph::opset7::Transpose>(leading_transpose), {0, 3, 1, 2}))
        return false;

    if (!TransposeOrderMatches(std::dynamic_pointer_cast<ngraph::opset7::Transpose>(trailing_transpose), {0, 2, 3, 1}))
        return false;

    if (bias && !VerifyBias(std::dynamic_pointer_cast<ngraph::opset7::Add>(bias), conv_data.filter_count))
        return false;

    GeneratePadding(std::dynamic_pointer_cast<ngraph::opset7::Transpose>(leading_transpose),
        std::dynamic_pointer_cast<ngraph::opset7::Convolution>(conv), conv_data);

    return true;
}

std::function<bool(ngraph::Output<ngraph::Node>)> consumers_and_rank(const size_t expected_count, const ngraph::Dimension& expected_rank) {
    return [=](ngraph::Output<ngraph::Node> output) -> bool {
        return ngraph::pattern::consumers_count(expected_count) && ngraph::pattern::rank_equals(expected_rank);
    };
}

ConvertPadded2ValidConv::ConvertPadded2ValidConv() {
    MATCHER_SCOPE(ConvertPadded2ValidConv);

    auto const_input = ngraph::pattern::wrap_type<ngraph::opset7::Constant>();
    auto leading_transpose = ngraph::pattern::wrap_type<ngraph::opset7::Transpose>({ngraph::pattern::any_input(), const_input},
        consumers_and_rank(1, 4));
    auto conv = ngraph::pattern::wrap_type<ngraph::opset7::Convolution>(
        {leading_transpose, ngraph::pattern::wrap_type<ngraph::opset7::Constant, ngraph::opset7::FakeQuantize>(ngraph::pattern::rank_equals(4))},
        ngraph::pattern::consumers_count(1));
    auto bias = ngraph::pattern::wrap_type<ngraph::opset7::Add>({conv, const_input},
        ngraph::pattern::consumers_count(1));
    auto fq = ngraph::pattern::wrap_type<ngraph::opset7::FakeQuantize>({bias, const_input, const_input, const_input, const_input},
        ngraph::pattern::consumers_count(1));
    auto max_pool1 = ngraph::pattern::wrap_type<ngraph::opset7::MaxPool>({bias},
        ngraph::pattern::consumers_count(1));
    auto max_pool2 = ngraph::pattern::wrap_type<ngraph::opset7::MaxPool>({fq},
        ngraph::pattern::consumers_count(1));
    auto af1 = ngraph::pattern::wrap_type<ngraph::opset7::Relu, ngraph::opset7::Sigmoid,
        ngraph::opset7::Tanh, ngraph::opset7::Abs, ngraph::opset7::Log, ngraph::opset7::Exp,
        ngraph::opset7::Sign, ngraph::opset7::Clamp>({bias}, ngraph::pattern::consumers_count(1));
    auto af2 = ngraph::pattern::wrap_type<ngraph::opset7::Relu, ngraph::opset7::Sigmoid,
        ngraph::opset7::Tanh, ngraph::opset7::Abs, ngraph::opset7::Log, ngraph::opset7::Exp,
        ngraph::opset7::Sign, ngraph::opset7::Clamp>({fq}, ngraph::pattern::consumers_count(1));
    auto af3 = ngraph::pattern::wrap_type<ngraph::opset7::Relu, ngraph::opset7::Sigmoid,
        ngraph::opset7::Tanh, ngraph::opset7::Abs, ngraph::opset7::Log, ngraph::opset7::Exp,
        ngraph::opset7::Sign, ngraph::opset7::Clamp>({max_pool1}, ngraph::pattern::consumers_count(1));
    auto af4 = ngraph::pattern::wrap_type<ngraph::opset7::Relu, ngraph::opset7::Sigmoid,
        ngraph::opset7::Tanh, ngraph::opset7::Abs, ngraph::opset7::Log, ngraph::opset7::Exp,
        ngraph::opset7::Sign, ngraph::opset7::Clamp>({max_pool2}, ngraph::pattern::consumers_count(1));
    auto transpose_input = std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{conv, bias, max_pool1, max_pool2, fq, af1, af2, af3, af4});
    auto trailing_transpose = ngraph::pattern::wrap_type<ngraph::opset7::Transpose>({transpose_input, const_input},
        consumers_and_rank(1, 4));

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto conv_output = conv->output(0).get_node_shared_ptr();
        IE_ASSERT(conv_output != nullptr);

        auto bias_node = std::dynamic_pointer_cast<ngraph::opset7::Add>(conv_output);

        return Convert(pattern_map.at(leading_transpose).get_node_shared_ptr(), pattern_map.at(conv).get_node_shared_ptr(),
            pattern_map.at(trailing_transpose).get_node_shared_ptr(), bias_node);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(trailing_transpose, matcher_name);
    this->register_matcher(m, callback);
}
