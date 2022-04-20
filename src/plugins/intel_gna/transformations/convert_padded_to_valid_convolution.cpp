// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/cc/ngraph/itt.hpp>

#include "transformations/convert_padded_to_valid_convolution.hpp"

#include <memory>

#include <ngraph/opsets/opset7.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pattern/op/or.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pass/manager.hpp>
#include <ie_common.h>
#include "utils/transformation_helper.hpp"


using namespace GNAPluginNS;

static bool VerifyAndGetConvData(std::shared_ptr<ngraph::opset7::Convolution> conv, ConvData& conv_data) {
    const auto& input = conv->input_value(0);

    // We support only batch 1
    if (input.get_shape()[0] != 1) {
        return false;
    }

    GetConvData(conv, conv_data);

    return conv_data.pads_begin_height || conv_data.pads_end_height || conv_data.pads_begin_width || conv_data.pads_end_width;
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

    if (!VerifyAndGetConvData(std::dynamic_pointer_cast<ngraph::opset7::Convolution>(conv), conv_data))
        return false;

    // We are looking for Transpose(NHWC->NCHW) => Conv => Transpose(NCHW->NHWC)
    // or similar cases, so required network must be in NHWC order like in TF
    if (!TransposeOrderMatches(std::dynamic_pointer_cast<ngraph::opset7::Transpose>(leading_transpose), {0, 3, 1, 2}))
        return false;

    if (!TransposeOrderMatches(std::dynamic_pointer_cast<ngraph::opset7::Transpose>(trailing_transpose), {0, 2, 3, 1}))
        return false;

    GeneratePadding(std::dynamic_pointer_cast<ngraph::opset7::Transpose>(leading_transpose),
        std::dynamic_pointer_cast<ngraph::opset7::Convolution>(conv), conv_data);

    return true;
}

static std::function<bool(ngraph::Output<ngraph::Node>)> consumers_and_rank(const size_t expected_count, const ngraph::Dimension& expected_rank) {
    return [=](ngraph::Output<ngraph::Node> output) -> bool {
        return ngraph::pattern::consumers_count(expected_count) && ngraph::pattern::rank_equals(expected_rank);
    };
}

ConvertPaddedToValidConv::ConvertPaddedToValidConv() {
    MATCHER_SCOPE(ConvertPaddedToValidConv);

    auto const_input = ngraph::pattern::wrap_type<ngraph::opset7::Constant>();
    auto leading_transpose = ngraph::pattern::wrap_type<ngraph::opset7::Transpose>({ngraph::pattern::any_input(), const_input},
        consumers_and_rank(1, 4));
    auto conv = ngraph::pattern::wrap_type<ngraph::opset7::Convolution>(
        {leading_transpose, ngraph::pattern::wrap_type<ngraph::opset7::Constant, ngraph::opset7::FakeQuantize>(ngraph::pattern::rank_equals(4))},
        ngraph::pattern::consumers_count(1));
    auto bias = ngraph::pattern::wrap_type<ngraph::opset7::Add>({conv, const_input},
        ngraph::pattern::consumers_count(1));
    auto fq_bias = ngraph::pattern::wrap_type<ngraph::opset7::FakeQuantize>({bias, const_input, const_input, const_input, const_input},
        ngraph::pattern::consumers_count(1));
    auto max_pool1 = ngraph::pattern::wrap_type<ngraph::opset7::MaxPool>({bias},
        ngraph::pattern::consumers_count(1));
    auto max_pool2 = ngraph::pattern::wrap_type<ngraph::opset7::MaxPool>({fq_bias},
        ngraph::pattern::consumers_count(1));
    auto af1 = ngraph::pattern::wrap_type<ngraph::opset7::Relu, ngraph::opset7::Sigmoid,
        ngraph::opset7::Tanh, ngraph::opset7::Abs, ngraph::opset7::Log, ngraph::opset7::Exp,
        ngraph::opset7::Sign, ngraph::opset7::Clamp>({conv}, ngraph::pattern::consumers_count(1));
    auto af2 = ngraph::pattern::wrap_type<ngraph::opset7::Relu, ngraph::opset7::Sigmoid,
        ngraph::opset7::Tanh, ngraph::opset7::Abs, ngraph::opset7::Log, ngraph::opset7::Exp,
        ngraph::opset7::Sign, ngraph::opset7::Clamp>({bias}, ngraph::pattern::consumers_count(1));
    auto af3 = ngraph::pattern::wrap_type<ngraph::opset7::Relu, ngraph::opset7::Sigmoid,
        ngraph::opset7::Tanh, ngraph::opset7::Abs, ngraph::opset7::Log, ngraph::opset7::Exp,
        ngraph::opset7::Sign, ngraph::opset7::Clamp>({fq_bias}, ngraph::pattern::consumers_count(1));
    auto af4 = ngraph::pattern::wrap_type<ngraph::opset7::Relu, ngraph::opset7::Sigmoid,
        ngraph::opset7::Tanh, ngraph::opset7::Abs, ngraph::opset7::Log, ngraph::opset7::Exp,
        ngraph::opset7::Sign, ngraph::opset7::Clamp>({max_pool1}, ngraph::pattern::consumers_count(1));
    auto af5 = ngraph::pattern::wrap_type<ngraph::opset7::Relu, ngraph::opset7::Sigmoid,
        ngraph::opset7::Tanh, ngraph::opset7::Abs, ngraph::opset7::Log, ngraph::opset7::Exp,
        ngraph::opset7::Sign, ngraph::opset7::Clamp>({max_pool2}, ngraph::pattern::consumers_count(1));
    auto fq_af1 = ngraph::pattern::wrap_type<ngraph::opset7::FakeQuantize>({af3, const_input, const_input, const_input, const_input},
        ngraph::pattern::consumers_count(1));
    auto fq_af2 = ngraph::pattern::wrap_type<ngraph::opset7::FakeQuantize>({af5, const_input, const_input, const_input, const_input},
        ngraph::pattern::consumers_count(1));
    auto transpose_input =
        std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{conv, bias, max_pool1, max_pool2, fq_bias, af1, af2, af3, af4, af5, fq_af1, fq_af2});
    auto trailing_transpose = ngraph::pattern::wrap_type<ngraph::opset7::Transpose>({transpose_input, const_input},
        consumers_and_rank(1, 4));

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto bias_it = pattern_map.find(bias);
        auto bias_node = (bias_it == std::end(pattern_map) ? nullptr : bias_it->second.get_node_shared_ptr());

        if (bias_node && !VerifyBiasGetConst(pattern_map.at(conv).get_node_shared_ptr(), bias_node))
            return false;

        return Convert(pattern_map.at(leading_transpose).get_node_shared_ptr(), pattern_map.at(conv).get_node_shared_ptr(),
            pattern_map.at(trailing_transpose).get_node_shared_ptr(), bias_node);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(trailing_transpose, matcher_name);
    this->register_matcher(m, callback);
}
