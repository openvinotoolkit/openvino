// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_padded2valid_conv.hpp"

#include <memory>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/builder/reshape.hpp>
#include <ngraph/builder/split.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include "itt.hpp"

using namespace ngraph;
using namespace op;

static bool TransposeOrderMatches(std::shared_ptr<Transpose> transpose, std::vector<int64_t> order) {
    if (!transpose)
        return false;
    const Output<Node>& transpose_order = transpose->input_value(1);
    auto transpose_order_dim = transpose_order.get_shape().size();

    if (transpose_order_dim != 1 || transpose_order.get_shape()[0] != order.size())
        return false;

    auto const_with_order_values = std::dynamic_pointer_cast<ngraph::opset1::Constant>(transpose_order.get_node_shared_ptr());
    if (!const_with_order_values)
        return false;

    const int64_t* data = const_with_order_values->get_data_ptr<int64_t>();
    if (!data)
        return false;

    for (size_t i = 0; i < order.size(); i++) {
        if (order[i] != data[i])
            return false;
    }

    return true;
}

static std::shared_ptr<opset1::StridedSlice> FlatCrop(Output<Node> input, size_t offset, size_t size) {
    auto shape = input.get_shape();
    if (shape.size() == 1) {
        return std::make_shared<ngraph::opset1::StridedSlice>(
            input, // data
            ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{ 1 }, { offset }), // begin slice index
            ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{ 1 }, { offset + size }), // end slice index
            ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{ 1 }, { 1 }), // strides
            std::vector<int64_t>{0},  // begin mask
            std::vector<int64_t>{0}); // end mask
    } else if (shape.size() == 2) {
        return std::make_shared<ngraph::opset1::StridedSlice>(
            input, // data
            ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{ 2 }, { (size_t)0, offset }), // begin sice index
            ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{ 2 }, { (size_t)0, offset + size }), // end slice index
            ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{ 2 }, { (size_t)1, (size_t)1 }), // strides
            std::vector<int64_t>{1, 0},  // begin mask
            std::vector<int64_t>{1, 0}); // end mask
    }
    return nullptr;
}

NGRAPH_RTTI_DEFINITION(ngraph::pass::ConvertPadded2ValidConv, "ConvertPadded2ValidConv", 0);
bool ngraph::pass::ConvertPadded2ValidConv::run_on_function(std::shared_ptr<ngraph::Function> f) {
    // Traverse nGraph Function in topological order
    bool is_graph_modfied = false;
    for (auto& node : f->get_ordered_ops()) {
        auto conv = std::dynamic_pointer_cast<ngraph::opset1::Convolution> (node);
        if (nullptr == conv || transformation_callback(conv)) {
            continue;
        }

        const Output<Node>& input = conv->input_value(0);
        const Output<Node>& filters = conv->input_value(1);
        auto output_shape = conv->get_output_shape(0);
        auto padding_type = conv->get_auto_pad();

        // we support only 2D conv batch 1
        if (input.get_shape().size() != 4 ||
            filters.get_shape().size() != 4 ||
            output_shape.size() != 4 ||
            conv->get_dilations().size() != 2 ||
            conv->get_strides().size() != 2 ||
            input.get_shape()[0] != 1) {
            continue;
        }
        // we are looking for Transpose(NHWC->NCHW) => conv => Transpose(NCHW->NHWC)
        // so required network must be in NHWC order like in TF
        //   supported cases:
        //     - Transpose(NHWC->NCHW) => conv => Transpose(NCHW->NHWC)
        //     - Transpose(NHWC->NCHW) => conv => broadcasted add (BIAS) => Transpose(NCHW->NHWC)
        //     - Transpose(NHWC->NCHW) => conv => broadcasted add (BIAS) => MaxPooling => Transpose(NCHW->NHWC) (2d max pool case)
        //     - Transpose(NHWC->NCHW) => conv => broadcasted add (BIAS) => ActivationFunction => Transpose(NCHW->NHWC)
        //     - Transpose(NHWC->NCHW) => conv => broadcasted add (BIAS) => MaxPool => ActivationFunction => Transpose(NCHW->NHWC)
        //     - Transpose(NHWC->NCHW) => conv => Transpose(NCHW->NHWC) => BIAS (output of MO --disable_nhwc_to_nchw option)
        //     - Transpose(NHWC->NCHW) => conv => Transpose(NCHW->NHWC) => BIAS => AF (output of MO --disable_nhwc_to_nchw option)
        auto leading_transpose = std::dynamic_pointer_cast<Transpose>(input.get_node_shared_ptr());
        if (!leading_transpose || !TransposeOrderMatches(leading_transpose, { 0, 3, 1, 2 }))
            continue;

        // check if convolution output port is connected with only one Op
        auto output_0 = node->get_output_target_inputs(0);
        if (output_0.size() != 1)
            continue;

        auto filter_values = std::dynamic_pointer_cast<ngraph::opset1::Constant>(filters.get_node_shared_ptr());
        if (!filter_values) {
            continue;
        }
        size_t input_channel_count = input.get_shape()[1];
        size_t input_height = input.get_shape()[2];
        size_t input_width = input.get_shape()[3];

        size_t filter_count = filters.get_shape()[0];

        size_t filter_height = filters.get_shape()[2];
        size_t filter_width = filters.get_shape()[3];

        auto output_0_node = output_0.begin()->get_node()->shared_from_this();
        auto trailing_transpose = std::dynamic_pointer_cast<Transpose>(output_0_node);
        auto conv_bias = std::dynamic_pointer_cast<ngraph::opset1::Add>(output_0_node);
        auto max_pool = std::dynamic_pointer_cast<ngraph::opset1::MaxPool>(output_0_node);
        auto af = std::dynamic_pointer_cast<ngraph::op::util::UnaryElementwiseArithmetic>(output_0_node);
        std::shared_ptr<Node>last_op_in_sequence_for_replacement = trailing_transpose;

        std::shared_ptr<ngraph::Node> bias_const;
        if (leading_transpose && trailing_transpose && conv) {
            auto trailing_transpose_output_0 = trailing_transpose->get_output_target_inputs(0);
            if (trailing_transpose_output_0.size() == 1) {
                auto trailing_transpose_output_0_node = trailing_transpose_output_0.begin()->get_node()->shared_from_this();
                auto add_op = std::dynamic_pointer_cast<ngraph::opset1::Add>(trailing_transpose_output_0_node);
                max_pool = std::dynamic_pointer_cast<ngraph::opset1::MaxPool>(trailing_transpose_output_0_node);
                af = std::dynamic_pointer_cast<ngraph::op::util::UnaryElementwiseArithmetic>(trailing_transpose_output_0_node);
                if (add_op) {
                    auto add_const = std::dynamic_pointer_cast<ngraph::op::Constant>(add_op->input_value(1).get_node_shared_ptr());
                    if (add_const) {
                        auto bias_size = shape_size(add_const->get_shape());
                        // the add maybe normal add not bias, than we just go further
                        if (bias_size == filter_count) {
                            conv_bias = add_op;
                            last_op_in_sequence_for_replacement = add_op;

                            auto bias_output_0 = add_op->get_output_target_inputs(0);
                            if (bias_output_0.size() == 1) {
                                auto bias_output_0_node = bias_output_0.begin()->get_node()->shared_from_this();
                                max_pool = std::dynamic_pointer_cast<ngraph::opset1::MaxPool>(bias_output_0_node);
                                af = std::dynamic_pointer_cast<ngraph::op::util::UnaryElementwiseArithmetic>(bias_output_0_node);
                            }
                        }
                    }
                }
            }
        } else if (!trailing_transpose && conv_bias) {
            // the NCHW order
            auto bias_output_0 = conv_bias->get_output_target_inputs(0);
            if (bias_output_0.size() != 1)
                continue;

            auto bias_output_0_node = bias_output_0.begin()->get_node()->shared_from_this();
            trailing_transpose = std::dynamic_pointer_cast<Transpose>(bias_output_0_node);
            last_op_in_sequence_for_replacement = trailing_transpose;
            max_pool = std::dynamic_pointer_cast<ngraph::opset1::MaxPool>(bias_output_0_node);
            af = std::dynamic_pointer_cast<ngraph::op::util::UnaryElementwiseArithmetic>(bias_output_0_node);
        }

        if (max_pool) {
            auto maxpool_output_0 = max_pool->get_output_target_inputs(0);
            if (maxpool_output_0.size() != 1)
                continue;
            auto maxpool_output_0_node = maxpool_output_0.begin()->get_node()->shared_from_this();
            // disable_nhwc_to_nchw option case
            if (!trailing_transpose) {
                trailing_transpose = std::dynamic_pointer_cast<Transpose>(maxpool_output_0_node);
                last_op_in_sequence_for_replacement = trailing_transpose;
            } else {
                last_op_in_sequence_for_replacement = max_pool;
            }
            af = std::dynamic_pointer_cast<ngraph::op::util::UnaryElementwiseArithmetic>(maxpool_output_0_node);
        }

        //and finally activation function
        if (af) {
            auto af_output_0 = af->get_output_target_inputs(0);
            if (af_output_0.size() != 1)
                continue;
            auto af_output_0_node = af_output_0.begin()->get_node()->shared_from_this();
            if (!trailing_transpose) {
                trailing_transpose = std::dynamic_pointer_cast<Transpose>(af_output_0_node);
                last_op_in_sequence_for_replacement = trailing_transpose;
            } else {
                last_op_in_sequence_for_replacement = af;
            }
        }

        if (!last_op_in_sequence_for_replacement || !trailing_transpose || !TransposeOrderMatches(trailing_transpose, { 0, 2, 3, 1 }))
            continue;

        size_t filter_dilation_x = conv->get_dilations()[1];
        size_t filter_dilation_y = conv->get_dilations()[0];

        size_t filter_stride_x = conv->get_strides()[1];
        size_t filter_stride_y = conv->get_strides()[0];

        // we are assuming VALID conv
        size_t pads_begin_x = 0;
        size_t pads_begin_y = 0;
        size_t pads_end_x = 0;
        size_t pads_end_y = 0;

        size_t output_channel_count = filter_count;
        size_t output_height = 0;
        size_t output_width = 0;

        switch (padding_type) {
        case ngraph::op::PadType::EXPLICIT:
            pads_begin_y = conv->get_pads_begin()[0];
            pads_begin_x = conv->get_pads_begin()[1];
            pads_end_y = conv->get_pads_end()[0];
            pads_end_x = conv->get_pads_end()[1];
            break;
        case ngraph::op::PadType::VALID:
            // all padding equal to 0 - already set
            break;
        case ngraph::op::PadType::SAME_LOWER:
        case ngraph::op::PadType::SAME_UPPER:
        {
            output_height = output_shape[2];
            output_width = output_shape[3];

            size_t pad_begin_n_end_y = output_height * filter_stride_y + (filter_height)*filter_dilation_y - input_height - 1;
            size_t pad_begin_n_end_x = output_width * filter_stride_x + (filter_width)*filter_dilation_x - input_width - 1;
            pads_begin_y = (ngraph::op::PadType::SAME_LOWER == padding_type) ? (pad_begin_n_end_y >> 1) + (pad_begin_n_end_y & 1) : (pad_begin_n_end_y >> 1);
            pads_end_y = (ngraph::op::PadType::SAME_UPPER == padding_type) ? (pad_begin_n_end_y >> 1) + (pad_begin_n_end_y & 1) : (pad_begin_n_end_y >> 1);
            pads_begin_x = (ngraph::op::PadType::SAME_LOWER == padding_type) ? (pad_begin_n_end_x >> 1) + (pad_begin_n_end_x & 1) : (pad_begin_n_end_x >> 1);
            pads_end_x = (ngraph::op::PadType::SAME_UPPER == padding_type) ? (pad_begin_n_end_x >> 1) + (pad_begin_n_end_x & 1) : (pad_begin_n_end_x >> 1);

            break;
        }
        default:
            break;
        }
        output_height = (input_height + pads_begin_y + pads_end_y - ((filter_height - 1) * filter_dilation_y + 1)) / filter_stride_y + 1;
        output_width = (input_width + pads_begin_x + pads_end_x - ((filter_width - 1) * filter_dilation_x + 1)) / filter_stride_x + 1;

        if (output_channel_count != output_shape[1] ||
            output_height != output_shape[2] ||
            output_width != output_shape[3]) {
            continue;
        }

        // No padding - there is no need to decompose such convolution
        if (pads_begin_y == 0 && pads_end_y == 0 && pads_begin_x == 0 && pads_end_x == 0)
            continue;

        // All checks applied - now we may start to do transformations

        size_t flat_left_padding = input_channel_count * pads_begin_x;
        size_t flat_right_padding = input_channel_count * pads_end_x;
        size_t flat_top_padding = input_channel_count * (pads_begin_x + input_width + pads_end_x) * pads_begin_y;
        size_t flat_bottom_padding = input_channel_count * (pads_begin_x + input_width + pads_end_x) * pads_end_y;
        size_t biggest_padding = std::max(std::max(flat_left_padding, flat_right_padding), std::max(flat_top_padding, flat_bottom_padding));
        size_t padded_row_size = input_channel_count * (pads_begin_x + input_width + pads_end_x);

        if (input_height > 1 && (flat_top_padding > 1 || flat_bottom_padding > 1)) {
            biggest_padding = biggest_padding > padded_row_size ? biggest_padding : padded_row_size;
        }

        auto flat_input = builder::opset1::reshape(
            leading_transpose->input_value(0),
            Shape{ (size_t)1, shape_size(leading_transpose->input_value(0).get_shape()) });
        // zero padding
        auto const_holding_padding = std::make_shared<opset1::Constant>(element::Type_t::f32, Shape{ 1, biggest_padding }, 0);

        // padding
        // padding
        // ... row ...
        // ... row ...
        // ...........
        // ... row ...
        // padding
        // padding

        // Add top padding
        OutputVector input_rows_to_concat;

        // padding
        for (size_t p = 0; p < pads_begin_y; p++) {
            if (padded_row_size == biggest_padding) {
                input_rows_to_concat.push_back(const_holding_padding);
            } else {
                auto slice = FlatCrop(const_holding_padding, 0, padded_row_size);
                ngraph::copy_runtime_info(conv, slice);
                input_rows_to_concat.push_back(slice);
            }
        }

        // pad every row of input plan
        for (size_t h = 0; h < input_height; h++) {
            // left padding     input     right padding
            //     |              |           |
            //     +--------------+-----------+
            //                    |
            //                 concat

            auto not_padded_row = input_height == 1 ?
                flat_input :
                FlatCrop(flat_input, h * input_width * input_channel_count, input_width * input_channel_count);
            ngraph::copy_runtime_info(conv, not_padded_row);
            if (flat_left_padding || flat_right_padding) {
                OutputVector single_row_concat_inputs;
                if (flat_left_padding) {
                    if (flat_left_padding == biggest_padding) {
                        single_row_concat_inputs.push_back(const_holding_padding);
                    } else {
                        auto slice = FlatCrop(const_holding_padding, 0, flat_left_padding);
                        ngraph::copy_runtime_info(conv, slice);
                        single_row_concat_inputs.push_back(slice);
                    }
                }
                single_row_concat_inputs.push_back(not_padded_row);
                if (flat_right_padding) {
                    if (flat_right_padding == biggest_padding) {
                        single_row_concat_inputs.push_back(const_holding_padding);
                    } else {
                        auto slice = FlatCrop(const_holding_padding, 0, flat_right_padding);
                        ngraph::copy_runtime_info(conv, slice);
                        single_row_concat_inputs.push_back(slice);
                    }
                }
                auto padded_row_concat = std::make_shared<opset1::Concat>(single_row_concat_inputs, 1);
                ngraph::copy_runtime_info(conv, padded_row_concat);
                input_rows_to_concat.push_back(padded_row_concat);
            } else {
                input_rows_to_concat.push_back(not_padded_row);
            }
        }
        // Bottom padding
        for (size_t p = 0; p < pads_end_y; p++) {
            if (padded_row_size == biggest_padding) {
                input_rows_to_concat.push_back(const_holding_padding);
            } else {
                auto slice = FlatCrop(const_holding_padding, 0, padded_row_size);
                ngraph::copy_runtime_info(conv, slice);
                input_rows_to_concat.push_back(slice);
            }
        }
        auto padded_input_plane = std::make_shared<opset1::Concat>(input_rows_to_concat, 1);
        ngraph::copy_runtime_info(conv, padded_input_plane);

        auto padded_input_plane_reshaped = builder::opset1::reshape(padded_input_plane,
            Shape{ 1, pads_begin_y + input_height + pads_end_y, pads_begin_x + input_width + pads_end_x, input_channel_count });
        //NHWC => NCHW
        auto transposed2chw = builder::opset1::reorder_axes(padded_input_plane_reshaped, { 0, 3, 1, 2 });

        auto conv_copy = std::make_shared<ngraph::opset1::Convolution>(
            transposed2chw->output(0),
            conv->input_value(1),
            conv->get_strides(),
            CoordinateDiff{ 0, 0 },
            CoordinateDiff{ 0, 0 },
            conv->get_dilations(),
            PadType::EXPLICIT);

        ngraph::replace_node(conv, conv_copy);

        is_graph_modfied = true;
    }
    return is_graph_modfied;
}
