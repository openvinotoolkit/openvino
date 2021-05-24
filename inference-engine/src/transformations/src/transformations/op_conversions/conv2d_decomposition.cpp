// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/conv2d_decomposition.hpp"
#include "transformations/utils/utils.hpp"

#include <memory>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/builder/reshape.hpp>
#include <ngraph/builder/split.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph_ops/type_relaxed.hpp>
#include "itt.hpp"


using namespace ngraph;
using namespace op;

#define GNA_MAX_1D_CONV_CHANNEL_COUNT 768
#define GNA_MAX_PERMUTE_COL_COUNT 8

static std::vector<std::shared_ptr<opset1::Constant>> ReduceConv2DFilterHeightByChannelPermute(std::shared_ptr<opset1::Constant>& filters,
                                                                                        bool vertical_permute, bool horizontal_permute, size_t split_channels) {
    std::vector <std::shared_ptr<opset1::Constant>> result;
    auto filter_shape = filters->get_output_shape(0);
    if (!horizontal_permute && !vertical_permute && split_channels == 1)
        return { filters };

    if (filter_shape.size() == 4) {
        std::vector<std::vector<float>> flat_filters;
        flat_filters.resize(split_channels);
        for (size_t i=0; i < split_channels; i++)
            flat_filters[i].resize(shape_size(filter_shape)/split_channels);

        auto N = filter_shape[0];
        auto C = filter_shape[1];
        auto H = filter_shape[2];
        auto W = filter_shape[3];

        size_t CS = (C / split_channels);
        const float* data = filters->get_data_ptr<float>();
        if (!(vertical_permute ^ horizontal_permute) || (vertical_permute && (!horizontal_permute))) {
            for (size_t n = 0; n < N; n++) {
                for (size_t c = 0; c < CS; c++) {
                    for (size_t s = 0; s < split_channels; s++) {
                        for (size_t h = 0; h < H; h++) {
                            for (size_t w = 0; w < W; w++) {
                                flat_filters[s][n * CS * H * W + c * H * W + h * W + w] = data[n * C * H * W + (c*split_channels+s) * H * W + h * W + w];
                            }
                        }
                    }
                }
            }
        } else if (vertical_permute) {
            for (size_t n = 0; n < N; n++) {
                for (size_t c = 0; c < CS; c++) {
                    for (size_t s = 0; s < split_channels; s++) {
                        for (size_t h = 0; h < H; h++) {
                            for (size_t w = 0; w < W; w++) {
                                flat_filters[s][n * CS * H * W + c * H * W + w * H + h] = data[n * C * H * W + (c * split_channels + s) * H * W + h * W + w];
                            }
                        }
                    }
                }
            }
        }
        if (vertical_permute && horizontal_permute) {
            for (auto new_filter : flat_filters)
                result.push_back(std::make_shared<opset1::Constant>(element::f32,
                    Shape{ filter_shape[0], filter_shape[1] * filter_shape[2] * filter_shape[3] / split_channels, 1, 1 }, new_filter));
        } else if (vertical_permute && !horizontal_permute) {
            for (auto new_filter : flat_filters)
                result.push_back(std::make_shared<opset1::Constant>(element::f32,
                    Shape{ filter_shape[0], filter_shape[1] * filter_shape[2] / split_channels, 1, filter_shape[3] }, new_filter));
        } else {
            for (auto new_filter : flat_filters)
                result.push_back(std::make_shared<opset1::Constant>(element::f32,
                    Shape{ filter_shape[0], filter_shape[1] / split_channels, filter_shape[2], filter_shape[3] }, new_filter));
        }
    }

    return result;
}

static std::shared_ptr<opset1::StridedSlice> FlatCrop(Output<Node> input, size_t offset, size_t size) {
    auto shape = input.get_shape();
    if (shape.size() == 1) {
        return std::make_shared<opset1::StridedSlice>(
            input, // data
            opset1::Constant::create(element::i64, Shape{ 1 }, { offset }), // begin slice index
            opset1::Constant::create(element::i64, Shape{ 1 }, { offset + size }), // end slice index
            opset1::Constant::create(element::i64, Shape{ 1 }, { 1 }), // strides
            std::vector<int64_t>{0},  // begin mask
            std::vector<int64_t>{0}); // end mask
    } else if (shape.size() == 2) {
        return std::make_shared<opset1::StridedSlice>(
            input, // data
            opset1::Constant::create(element::i64, Shape{ 2 }, { (size_t)0, offset }), // begin sice index
            opset1::Constant::create(element::i64, Shape{ 2 }, { (size_t)0, offset + size }), // end slice index
            opset1::Constant::create(element::i64, Shape{ 2 }, { (size_t)1, (size_t)1 }), // strides
            std::vector<int64_t>{1, 0},  // begin mask
            std::vector<int64_t>{1, 0}); // end mask
    }
    return nullptr;
}

static bool TransposeOrderMatches(std::shared_ptr<Transpose> transpose, std::vector<int64_t> order) {
    if (!transpose)
        return false;
    const Output<Node>& transpose_order = transpose->input_value(1);
    auto transpose_order_dim = transpose_order.get_shape().size();

    if (transpose_order_dim != 1 || transpose_order.get_shape()[0] != order.size())
        return false;

    auto const_with_order_values = std::dynamic_pointer_cast<opset1::Constant>(transpose_order.get_node_shared_ptr());
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

NGRAPH_RTTI_DEFINITION(pass::Conv2dDecomposition, "Conv2dDecomposition", 0);
bool pass::Conv2dDecomposition::run_on_function(std::shared_ptr<Function> f) {
    // Traverse nGraph Function in topological order
    bool is_graph_modfied = false;
    for (auto& node : f->get_ordered_ops()) {
        auto conv = std::dynamic_pointer_cast<opset1::Convolution> (node);
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
            input.get_shape()[0] != 1 ||
            filters.get_shape()[2] == 1 ||
            filters.get_shape()[3] == 1) {
            continue;
        }
        // TODO: Check if filter weights are not dynamic
        // TODO: Check BIAS sizes

        // we are looking for Transpose(NHWC->NCHW) => conv => Transpose(NCHW->NHWC)
        // so required network must be in NHWC order like in TF
        //   supported cases:
        //     - Transpose(NHWC->NCHW) => conv => Transpose(NCHW->NHWC)
        //     - Transpose(NHWC->NCHW) => conv => broadcasted add (BIAS) => Transpose(NCHW->NHWC)
        //     - Transpose(NHWC->NCHW) => conv => broadcasted add (BIAS) => MaxPooling => Transpose(NCHW->NHWC) (2d max pool case)
        //          ( TODO: 2d max pool case )
        //     - Transpose(NHWC->NCHW) => conv => broadcasted add (BIAS) => ActivationFunction => Transpose(NCHW->NHWC)
        //     - Transpose(NHWC->NCHW) => conv => broadcasted add (BIAS) => MaxPool => ActivationFunction => Transpose(NCHW->NHWC)
        //          ( TODO: 2d max pool case )
        //     - Transpose(NHWC->NCHW) => conv => Transpose(NCHW->NHWC) => BIAS (output of MO --disable_nhwc_to_nchw option)
        //     - Transpose(NHWC->NCHW) => conv => Transpose(NCHW->NHWC) => BIAS => AF (output of MO --disable_nhwc_to_nchw option)
        auto leading_transpose = std::dynamic_pointer_cast<Transpose>(input.get_node_shared_ptr());
        if (!leading_transpose || !TransposeOrderMatches(leading_transpose, {0, 3, 1, 2}))
            continue;

        // check if convolution output port is connected with only one Op
        auto output_0 = node->get_output_target_inputs(0);
        if (output_0.size() != 1)
            continue;

        auto filter_values = std::dynamic_pointer_cast<opset1::Constant>(filters.get_node_shared_ptr());
        if (!filter_values) {
            continue;
        }
        size_t input_channel_count = input.get_shape()[1];
        size_t input_height = input.get_shape()[2];
        size_t input_width = input.get_shape()[3];

        size_t filter_count = filters.get_shape()[0];
        size_t filter_height = filters.get_shape()[2];
        size_t filter_width = filters.get_shape()[3];

        if (filter_width > GNA_MAX_PERMUTE_COL_COUNT || filter_height > GNA_MAX_PERMUTE_COL_COUNT) {
            continue;
        }

        auto output_0_node = output_0.begin()->get_node()->shared_from_this();
        auto trailing_transpose = std::dynamic_pointer_cast<Transpose>(output_0_node);
        auto conv_bias = std::dynamic_pointer_cast<opset1::Add>(output_0_node);
        auto max_pool = std::dynamic_pointer_cast<opset1::MaxPool>(output_0_node);
        auto af = std::dynamic_pointer_cast<op::util::UnaryElementwiseArithmetic>(output_0_node);
        std::shared_ptr<Node>last_op_in_sequence_for_replacement = trailing_transpose;

        std::shared_ptr<Node> bias_const;
        bool disable_nhwc_to_nchw_option = false;
        if (leading_transpose && trailing_transpose && conv) {
            auto trailing_transpose_output_0 = trailing_transpose->get_output_target_inputs(0);
            if (trailing_transpose_output_0.size() == 1) {
                auto trailing_transpose_output_0_node = trailing_transpose_output_0.begin()->get_node()->shared_from_this();
                auto add_op = std::dynamic_pointer_cast<opset1::Add>(trailing_transpose_output_0_node);
                max_pool = std::dynamic_pointer_cast<opset1::MaxPool>(trailing_transpose_output_0_node);
                af = std::dynamic_pointer_cast<op::util::UnaryElementwiseArithmetic>(trailing_transpose_output_0_node);
                if (add_op) {
                    auto add_const = std::dynamic_pointer_cast<op::Constant>(add_op->input_value(1).get_node_shared_ptr());
                    if (add_const) {
                        auto bias_size = shape_size(add_const->get_shape());
                        // the add maybe normal add not bias, than we just go further
                        if (bias_size == filter_count) {
                            conv_bias = add_op;
                            last_op_in_sequence_for_replacement = add_op;
                            disable_nhwc_to_nchw_option = true;

                            auto bias_output_0 = add_op->get_output_target_inputs(0);
                            if (bias_output_0.size() == 1) {
                                auto bias_output_0_node = bias_output_0.begin()->get_node()->shared_from_this();
                                max_pool = std::dynamic_pointer_cast<opset1::MaxPool>(bias_output_0_node);
                                af = std::dynamic_pointer_cast<op::util::UnaryElementwiseArithmetic>(bias_output_0_node);
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
            max_pool = std::dynamic_pointer_cast<opset1::MaxPool>(bias_output_0_node);
            af = std::dynamic_pointer_cast<op::util::UnaryElementwiseArithmetic>(bias_output_0_node);
        }

        size_t pool_size_x = 1;
        size_t pool_stride_x = 1;

        if (max_pool) {
            // Check if MaxPool vertical stride == pool size
            auto pool_strides = max_pool->get_strides();
            auto pool_kernel = max_pool->get_kernel();
            // We support only VALID PADDING
            if (max_pool->get_auto_pad() != PadType::VALID)
                continue;

            if (pool_kernel.size() != 2 || pool_strides.size() != 2)
                continue;

            if (pool_kernel[0] != pool_strides[0] || pool_kernel[0] > 8)
                continue;
            pool_size_x = pool_kernel[1];
            pool_stride_x = pool_strides[1];

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
                disable_nhwc_to_nchw_option = true;
            }
            af = std::dynamic_pointer_cast<op::util::UnaryElementwiseArithmetic>(maxpool_output_0_node);
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
                disable_nhwc_to_nchw_option = true;
            }
        }

        if (!last_op_in_sequence_for_replacement || !trailing_transpose || !TransposeOrderMatches(trailing_transpose, {0, 2, 3, 1}))
            continue;

        if (conv_bias) {
            auto add_const = std::dynamic_pointer_cast<op::Constant>(conv_bias->input_value(1).get_node_shared_ptr());
            if (add_const) {
                auto bias_size = shape_size(add_const->get_shape());
                if (bias_size == filter_count) {
                    const float* srd_data_pointer = add_const->get_data_ptr<float>();
                    std::vector<float> bias_values(srd_data_pointer, srd_data_pointer + bias_size);
                    bias_const = opset1::Constant::create(element::Type_t::f32, Shape{ 1, bias_size , 1, 1 }, bias_values);
                } else {
                    continue;
                }
            } else {
                // BIAS size does not match (or dynamic BIAS), can't convert such convolution
                continue;
            }
        }

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
        case op::PadType::EXPLICIT:
            pads_begin_y = conv->get_pads_begin()[0];
            pads_begin_x = conv->get_pads_begin()[1];
            pads_end_y = conv->get_pads_end()[0];
            pads_end_x = conv->get_pads_end()[1];
            break;
        case op::PadType::VALID:
            // all padding equal to 0 - already set
            break;
        case op::PadType::SAME_LOWER:
        case op::PadType::SAME_UPPER:
        {
            output_height = output_shape[2];
            output_width = output_shape[3];

            size_t pad_begin_n_end_y = output_height * filter_stride_y + filter_height * filter_dilation_y - input_height - 1;
            size_t pad_begin_n_end_x = output_width * filter_stride_x + filter_width * filter_dilation_x - input_width - 1;
            pads_begin_y = (op::PadType::SAME_LOWER == padding_type) ? (pad_begin_n_end_y >> 1) + (pad_begin_n_end_y & 1) : (pad_begin_n_end_y >> 1);
            pads_end_y = (op::PadType::SAME_UPPER == padding_type) ? (pad_begin_n_end_y >> 1) + (pad_begin_n_end_y & 1) : (pad_begin_n_end_y >> 1);
            pads_begin_x = (op::PadType::SAME_LOWER == padding_type) ? (pad_begin_n_end_x >> 1) + (pad_begin_n_end_x & 1) : (pad_begin_n_end_x >> 1);
            pads_end_x = (op::PadType::SAME_UPPER == padding_type) ? (pad_begin_n_end_x >> 1) + (pad_begin_n_end_x & 1) : (pad_begin_n_end_x >> 1);

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

        size_t conv_count = 1;
        // Last check GNA limitations of 768 filters
        size_t total_factorized_conv_channel_count = (input_channel_count * filter_height * filter_width);
        while (total_factorized_conv_channel_count / conv_count > GNA_MAX_1D_CONV_CHANNEL_COUNT || total_factorized_conv_channel_count % conv_count != 0)
            conv_count++;
        //LIMITATION: currently we are able to split only convolutions without pooling in horizontal dimention
        if (conv_count > GNA_MAX_PERMUTE_COL_COUNT || ((pool_size_x > 1 || pool_stride_x > 1) && conv_count > 1))
            continue;

        // GNA supported features - there is no need to decompose such convolution
        if (conv_count == 1 && (input_height == 1 || input_width == 1) && filter_dilation_x == 1 && filter_dilation_y == 1 && !disable_nhwc_to_nchw_option)
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

        auto flat_input = builder::opset1::reshape(leading_transpose->input_value(0),
            Shape{ (size_t)1, shape_size(leading_transpose->input_value(0).get_shape()) });
        // zero padding
        // TODO: find biggest padding in whole network
        auto const_holding_padding = std::make_shared<opset1::Constant>(element::Type_t::f32, Shape{ 1, biggest_padding }, 0);

        copy_runtime_info(conv, const_holding_padding);

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
        auto padded_input_plane = flat_input;
        // padding
        if (pads_begin_x || pads_end_x || pads_begin_y || pads_end_y) {
            for (size_t p = 0; p < pads_begin_y; p++) {
                if (padded_row_size == biggest_padding) {
                    input_rows_to_concat.push_back(const_holding_padding);
                } else {
                    auto slice = FlatCrop(const_holding_padding, 0, padded_row_size);
                    copy_runtime_info(conv, slice);
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

                auto not_padded_row = input_height == 1 ? flat_input :
                    FlatCrop(flat_input, h * input_width * input_channel_count, input_width * input_channel_count);
                copy_runtime_info(conv, not_padded_row);
                if (flat_left_padding || flat_right_padding) {
                    OutputVector single_row_concat_inputs;
                    if (flat_left_padding) {
                        if (flat_left_padding == biggest_padding) {
                            single_row_concat_inputs.push_back(const_holding_padding);
                        } else {
                            auto slice = FlatCrop(const_holding_padding, 0, flat_left_padding);
                            copy_runtime_info(conv, slice);
                            single_row_concat_inputs.push_back(slice);
                        }
                    }
                    single_row_concat_inputs.push_back(not_padded_row);
                    if (flat_right_padding) {
                        if (flat_right_padding == biggest_padding) {
                            single_row_concat_inputs.push_back(const_holding_padding);
                        } else {
                            auto slice = FlatCrop(const_holding_padding, 0, flat_right_padding);
                            copy_runtime_info(conv, slice);
                            single_row_concat_inputs.push_back(slice);
                        }
                    }
                    auto padded_row_concat = std::make_shared<opset1::Concat>(single_row_concat_inputs, 1);
                    copy_runtime_info(conv, padded_row_concat);
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
                    copy_runtime_info(conv, slice);
                    input_rows_to_concat.push_back(slice);
                }
            }
            padded_input_plane = std::make_shared<opset1::Concat>(input_rows_to_concat, 1);
            copy_runtime_info(conv, padded_input_plane);
        }

        OutputVector split_planes;
        if (conv_count > 1) {
            auto reshape_before_permute = builder::opset1::reshape(padded_input_plane,
                Shape{ shape_size(padded_input_plane->get_shape()) / conv_count, conv_count});
            auto permute_before_channel_wise_split = builder::opset1::reorder_axes(reshape_before_permute, { 1ull, 0ull});
            auto reshape_after_permute = builder::opset1::reshape(permute_before_channel_wise_split,
                Shape{ (size_t)conv_count, padded_input_plane->get_shape()[1] / conv_count });
            split_planes = builder::opset1::split(reshape_after_permute, conv_count, 0);
        } else {
            split_planes.push_back(padded_input_plane);
        }

        bool vertical_permute = (filter_height > 1);
        bool horizontal_permute = (filter_dilation_x > 1);
        std::vector<std::shared_ptr<opset1::Constant>> h_1_filters =
            ReduceConv2DFilterHeightByChannelPermute(filter_values, vertical_permute, horizontal_permute, conv_count);
        for (auto filter : h_1_filters)
            copy_runtime_info(conv, filter);

        // if we split input planes due to GNA limitation - we must sum their results
        std::vector<std::shared_ptr<Node>> partial_conv_results;
        input_channel_count /= conv_count;

        for (size_t conv_index = 0; conv_index < conv_count; conv_index++) {
            Output<Node> reduced_input_plane = split_planes[conv_index];
            // lets change filter height to 1
            if (filter_height > 1) {
                /*
                 *              padded row - NHWC order
                 *                  |
                 *        split in vertical dim (filter height)
                 *                / | \
                 *                concat
                 *                  |
                 *                permute
                 */
                OutputVector dilated_input_planes;
                for (size_t f_y = 0; f_y < filter_height; f_y++) {
                    size_t offset = f_y * filter_dilation_y * (pads_begin_x + input_width + pads_end_x) * input_channel_count;
                    // point wise convolutions - as many as output width
                    auto slice = FlatCrop(reduced_input_plane, offset, (pads_begin_x + input_width + pads_end_x) * input_channel_count * output_height);
                    copy_runtime_info(conv, slice);
                    dilated_input_planes.push_back(slice);
                }
                // now let's flatten kernel of convolution in vertical dimension
                // it is done by interleaving dilated input planes
                auto dilated_chunks_concat = std::make_shared<opset1::Concat>(dilated_input_planes, 0);

                auto permuted_dilated_chunks = builder::opset1::transpose(dilated_chunks_concat);
                // flatten
                auto flatten_dilated_permuted_input = builder::opset1::reshape(permuted_dilated_chunks,
                    Shape{ (size_t)1, (pads_begin_x + input_width + pads_end_x) * input_channel_count * output_height * filter_height });

                copy_runtime_info(conv, { dilated_chunks_concat, flatten_dilated_permuted_input, permuted_dilated_chunks });
                reduced_input_plane = flatten_dilated_permuted_input;
            }
            OutputVector result_chunks;
            std::shared_ptr<Node> last_op;
            size_t h_1_filter_channel_count = (input_channel_count * filter_height);

            for (size_t y = 0; y < output_height; y += filter_stride_y) {
                size_t offset = y * (pads_begin_x + input_width + pads_end_x) * h_1_filter_channel_count;
                auto row = (output_height == 1) ? reduced_input_plane :
                    FlatCrop(reduced_input_plane, offset, (pads_begin_x + input_width + pads_end_x) * h_1_filter_channel_count);
                /*
                 *              padded row
                 *                  |
                 *        ??? <dilation !=1> ???
                 *                  |
                 *         split in vertical dim
                 *                / | \
                 *                concat
                 *                  |
                 *               permute
                 *                  |
                 *           permute NHWC => NCHW
                 *                  |
                 *                conv 1D (BIAS|MaxPooling)
                 *                  |
                 *           permute NCHW => NHWC
                 */
                auto nhwc_conv_y_input = row;
                if (horizontal_permute) {
                    // split
                    OutputVector dilated_chunks;
                    for (size_t f_x = 0; f_x < filter_width; f_x++) {
                        size_t offset = f_x * filter_dilation_x * h_1_filter_channel_count;
                        // point wise convolutions - as many as output width
                        auto slice = FlatCrop(row, offset, h_1_filter_channel_count * output_width);
                        copy_runtime_info(conv, slice);
                        dilated_chunks.push_back(slice);
                    }
                    // concat
                    auto dilated_chunks_concat = std::make_shared<opset1::Concat>(dilated_chunks, 0);

                    // permute
                    auto permuted_dilated_chunks = builder::opset1::transpose(dilated_chunks_concat);

                    // flatten
                    auto flatten_dilated_conv_input = builder::opset1::reshape(permuted_dilated_chunks,
                        Shape{ (size_t)1, 1, output_width, h_1_filter_channel_count * filter_width });
                    copy_runtime_info(conv, { flatten_dilated_conv_input, permuted_dilated_chunks, dilated_chunks_concat });

                    nhwc_conv_y_input = flatten_dilated_conv_input;
                }
                // decomposed nhwc convolution
                auto nhwc_conv_1d = [](std::shared_ptr<Node> source_conv2d,
                    Output<Node> input,
                    std::shared_ptr<Node> filters,
                    std::shared_ptr<Node> add_bias_const,
                    size_t stride_x,
                    size_t pool_size_x,
                    size_t pool_stride_x,
                    RoundingType rounding_type,
                    std::shared_ptr<op::util::UnaryElementwiseArithmetic> af,
                    size_t h_index,
                    size_t c_index = 0) {
                        // valid 1D convolution wrapped with permutes NHWC => NCHW => conv => NCHW => NHWC
                        // NHWC => NCHW
                        auto nchw_input = builder::opset1::reorder_axes(input, { 0ull, 3ull, 1ull, 2ull });
                        // conv
                        auto conv = std::make_shared<opset1::Convolution>(nchw_input, filters,
                            Strides{ 1, stride_x }, CoordinateDiff{ 0, 0 }, CoordinateDiff{ 0, 0 }, Strides{ 1, 1 }, PadType::VALID);
                        std::string conv_name = source_conv2d->get_friendly_name() + "_H_" + std::to_string(h_index) + "_CH_" + std::to_string(c_index);
                        conv->set_friendly_name(conv_name);

                        std::shared_ptr<Node> last_conv_block_op = conv;
                        if (add_bias_const) {
                            last_conv_block_op = std::make_shared<opset1::Add>(conv, add_bias_const);
                            copy_runtime_info(source_conv2d, last_conv_block_op);
                        }
                        //add max pooling
                        if (pool_size_x > 1 || pool_stride_x > 1) {
                            auto max_pool_x = opset1::MaxPool(last_conv_block_op, { 1, pool_stride_x }, { 0, 0 }, { 0, 0 },
                                { 1, pool_size_x }, rounding_type, op::PadType::VALID);
                            max_pool_x.validate_and_infer_types();
                            last_conv_block_op = std::make_shared <opset1::MaxPool>(max_pool_x);
                        }
                        if (af) {
                            auto af_result = af->copy_with_new_inputs({ last_conv_block_op });
                            copy_runtime_info(conv, af_result);
                            last_conv_block_op = af_result;
                        }

                        // NCHW => NHWC
                        auto nhwc_output = builder::opset1::reorder_axes(last_conv_block_op, { 0ull, 2ull, 3ull, 1ull });
                        copy_runtime_info(source_conv2d, { nchw_input, conv, nhwc_output });
                        return nhwc_output;
                };
                // this is pointwise convolution
                if (!horizontal_permute) {
                    size_t padded_row_width = pads_begin_x + input_width + pads_end_x;
                    size_t padded_row_flat_width = shape_size(nhwc_conv_y_input.get_shape());
                    nhwc_conv_y_input = builder::opset1::reshape(nhwc_conv_y_input, { 1ull, 1ull, padded_row_width, padded_row_flat_width / padded_row_width });
                }

                // valid 1D convolution wrapped with permutes NHWC => NCHW => conv => NCHW => NHWC
                // activation function can be fused with convolution only if it is not split
                auto nhwc_y_output = nhwc_conv_1d(conv, nhwc_conv_y_input, h_1_filters[conv_index], conv_index ? nullptr : bias_const,
                    filter_stride_x, pool_size_x, pool_stride_x, max_pool ? max_pool->get_rounding_type() : RoundingType::FLOOR,
                    conv_count == 1 ? af : nullptr, y);
                result_chunks.push_back(nhwc_y_output);
                last_op = nhwc_y_output;
            }
            // Vertical dimemsion greater than 1
            if (result_chunks.size() > 1) {
                // concat in H dim
                // in NHWC index of H is 1
                auto concatenated_sub_results = std::make_shared<opset1::Concat>(result_chunks, 1);
                copy_runtime_info(conv, concatenated_sub_results);
                last_op = concatenated_sub_results;
            }
            partial_conv_results.push_back(last_op);
        }
        std::shared_ptr<Node> conv_result = partial_conv_results[0];
        for (size_t i = 1; i < partial_conv_results.size(); i++) {
            auto add_result = std::make_shared<opset1::Add>(partial_conv_results[i], conv_result);
            copy_runtime_info(conv, add_result);
            conv_result = add_result;
        }

        //TODO: maxpool 2d case
        //if (max_pool && (pool_size_y > 1 || pool_stride_y > 1)) {
        //}

        // activation function
        if (af && conv_count > 1) {
            auto af_result = af->copy_with_new_inputs({ conv_result });
            copy_runtime_info(conv, af_result);
            conv_result = af_result;
        }
        // we need to put friendly name, so the conv output can be used as network result
        std::string conv_result_name = trailing_transpose->get_friendly_name();
        replace_node(last_op_in_sequence_for_replacement, conv_result);
        conv_result->set_friendly_name(conv_result_name);

        is_graph_modfied = true;
    }
    return is_graph_modfied;
}
