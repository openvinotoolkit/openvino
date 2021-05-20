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

namespace {
struct GraphData {
    std::shared_ptr<opset1::Convolution> conv;
    std::shared_ptr<Transpose> leading_transpose;
    std::shared_ptr<Transpose> trailing_transpose;
    std::shared_ptr<opset1::MaxPool> max_pool;
    std::shared_ptr<op::util::UnaryElementwiseArithmetic> af;
    std::shared_ptr<Node> bias_const;
    std::shared_ptr<Node>last_op_in_sequence_for_replacement;
    bool disable_nhwc_to_nchw_option;
};

struct ConvData {
    size_t input_height;
    size_t input_width;
    size_t input_channel_count;
    size_t filter_height;
    size_t filter_width;
    size_t filter_count;
    size_t filter_dilation_x;
    size_t filter_dilation_y;
    size_t filter_stride_x;
    size_t filter_stride_y;
    size_t pads_begin_y;
    size_t pads_begin_x;
    size_t pads_end_y;
    size_t pads_end_x;
    ngraph::op::PadType padding_type;
    size_t output_channel_count;
    ngraph::Shape output_shape;
};

struct MaxPoolData {
    size_t pool_size_x;
    size_t pool_stride_x;
    // TODO: currently 2d max pool is not supported
    //size_t pool_size_y;
    //size_t pool_stride_y;
};

struct OutData {
    size_t output_height;
    size_t output_width;
    size_t conv_count;
    std::shared_ptr<Node> padded_input_plane;
};

std::vector<std::shared_ptr<opset1::Constant>> ReduceConv2DFilterHeightByChannelPermute(std::shared_ptr<opset1::Constant>& filters,
    bool vertical_permute, bool horizontal_permute, size_t split_channels) {
    std::vector <std::shared_ptr<opset1::Constant>> result;
    auto filter_shape = filters->get_output_shape(0);
    if (!horizontal_permute && !vertical_permute && split_channels == 1)
        return { filters };

    if (filter_shape.size() == 4) {
        std::vector<std::vector<float>> flat_filters;
        flat_filters.resize(split_channels);
        for (size_t i = 0; i < split_channels; i++)
            flat_filters[i].resize(shape_size(filter_shape) / split_channels);

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
                                flat_filters[s][n * CS * H * W + c * H * W + h * W + w] =
                                    data[n * C * H * W + (c * split_channels + s) * H * W + h * W + w];
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
                                flat_filters[s][n * CS * H * W + c * H * W + w * H + h] =
                                    data[n * C * H * W + (c * split_channels + s) * H * W + h * W + w];
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

std::shared_ptr<opset1::StridedSlice> FlatCrop(Output<Node> input, size_t offset, size_t size) {
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

bool TransposeOrderMatches(std::shared_ptr<Transpose> transpose, std::vector<int64_t> order) {
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

void FillConvData(std::shared_ptr<opset1::Convolution> conv, ConvData &conv_data) {
    conv_data.output_shape = conv->get_output_shape(0);
    conv_data.padding_type = conv->get_auto_pad();
    conv_data.input_height = conv->input_value(0).get_shape()[2];
    conv_data.input_width = conv->input_value(0).get_shape()[3];
    conv_data.input_channel_count = conv->input_value(0).get_shape()[1];
    conv_data.filter_height = conv->input_value(1).get_shape()[2];
    conv_data.filter_width = conv->input_value(1).get_shape()[3];
    conv_data.filter_count = conv->input_value(1).get_shape()[0];
    conv_data.filter_dilation_x = conv->get_dilations()[1];
    conv_data.filter_dilation_y = conv->get_dilations()[0];
    conv_data.filter_stride_x = conv->get_strides()[1];
    conv_data.filter_stride_y = conv->get_strides()[0];
    conv_data.pads_begin_y = conv->get_pads_begin()[0];
    conv_data.pads_begin_x = conv->get_pads_begin()[1];
    conv_data.pads_end_y = conv->get_pads_end()[0];
    conv_data.pads_end_x = conv->get_pads_end()[1];
    conv_data.output_channel_count = conv_data.filter_count;
}

template <class T>
std::shared_ptr<T> DetectNextLayer(std::shared_ptr<Node> node) {
    auto output_0_node = node->get_output_target_inputs(0).begin()->get_node()->shared_from_this();
    return std::dynamic_pointer_cast<T>(output_0_node);
}

bool VerifyLayer(std::shared_ptr<Node> layer) {
    auto layer_output_0 = layer->get_output_target_inputs(0);
    return layer_output_0.size() == 1 ? true : false;
}

std::shared_ptr<opset1::Convolution> DetectVerifyConvolution(std::shared_ptr<Node> node) {
    auto conv = std::dynamic_pointer_cast<opset1::Convolution>(node);

    if (conv) {
        // check if convolution output port is connected with only one Op
        if (!VerifyLayer(node))
            return nullptr;

        const Output<Node>& input = conv->input_value(0);
        const Output<Node>& filters = conv->input_value(1);
        auto output_shape = conv->get_output_shape(0);

        if (!std::dynamic_pointer_cast<opset1::Constant>(filters.get_node_shared_ptr()))
            return nullptr;

        // we support only 2D conv batch 1
        if (input.get_shape().size() != 4 ||
            filters.get_shape().size() != 4 ||
            output_shape.size() != 4 ||
            conv->get_dilations().size() != 2 ||
            conv->get_strides().size() != 2 ||
            input.get_shape()[0] != 1 ||
            filters.get_shape()[2] == 1 ||
            filters.get_shape()[3] == 1) {
            return nullptr;
        }

        size_t filter_height = filters.get_shape()[2];
        size_t filter_width = filters.get_shape()[3];

        if (filter_width > GNA_MAX_PERMUTE_COL_COUNT || filter_height > GNA_MAX_PERMUTE_COL_COUNT) {
            return nullptr;
        }
    }
    return conv;
}

std::shared_ptr<Transpose> DetectVerifyLeadingTranspose(std::shared_ptr<opset1::Convolution> conv) {
    const Output<Node>& input = conv->input_value(0);
    auto leading_transpose = std::dynamic_pointer_cast<Transpose>(input.get_node_shared_ptr());

    if (!leading_transpose || !TransposeOrderMatches(leading_transpose, { 0, 3, 1, 2 }))
        return nullptr;

    return leading_transpose;
}

std::shared_ptr<Node> CreateBiasConst(std::shared_ptr<opset1::Add> conv_bias, const ConvData& conv_data) {
    auto add_const = std::dynamic_pointer_cast<op::Constant>(conv_bias->input_value(1).get_node_shared_ptr());

    if (add_const) {
        auto bias_size = shape_size(add_const->get_shape());

        // the add may be a normal add not bias, than we just go further
        if (bias_size == conv_data.filter_count) {
            const float* srd_data_pointer = add_const->get_data_ptr<float>();
            std::vector<float> bias_values(srd_data_pointer, srd_data_pointer + bias_size);
            return opset1::Constant::create(element::Type_t::f32, Shape{ 1, bias_size , 1, 1 }, bias_values);
        }
    }
    // BIAS size does not match (or dynamic BIAS), can't convert such convolution
    return nullptr;
}

bool VerifyMaxPool(std::shared_ptr<opset1::MaxPool> max_pool, MaxPoolData& pool_data) {
    // Check if MaxPool vertical stride == pool size
    auto pool_strides = max_pool->get_strides();
    auto pool_kernel = max_pool->get_kernel();

    // We support only VALID PADDING
    if (max_pool->get_auto_pad() != PadType::VALID ||
        pool_kernel.size() != 2 || pool_strides.size() != 2 ||
        pool_kernel[0] != pool_strides[0] || pool_kernel[0] > 8 ||
        !VerifyLayer(max_pool))
        return false;

    pool_data.pool_size_x = pool_kernel[1];
    pool_data.pool_stride_x = pool_strides[1];
    return true;
}

bool DetectGraphSequence(GraphData& graph_data, const ConvData& conv_data, MaxPoolData& pool_data) {
    std::shared_ptr<opset1::Add> conv_bias;
    graph_data.last_op_in_sequence_for_replacement = graph_data.conv;
    graph_data.disable_nhwc_to_nchw_option = false;

    if ((graph_data.trailing_transpose = DetectNextLayer<Transpose>(graph_data.conv))) {
        graph_data.last_op_in_sequence_for_replacement = graph_data.trailing_transpose;

        if (VerifyLayer(graph_data.trailing_transpose) &&
            (conv_bias = DetectNextLayer<opset1::Add>(graph_data.trailing_transpose)) &&
            (graph_data.bias_const = CreateBiasConst(conv_bias, conv_data))) {
            graph_data.last_op_in_sequence_for_replacement = conv_bias;
            graph_data.disable_nhwc_to_nchw_option = true;
        }
    } else if ((conv_bias = DetectNextLayer<opset1::Add>(graph_data.conv))) {
        if (!VerifyLayer(conv_bias) || !(graph_data.bias_const = CreateBiasConst(conv_bias, conv_data)))
            return false;

        if ((graph_data.trailing_transpose = DetectNextLayer<Transpose>(conv_bias))) {
            graph_data.last_op_in_sequence_for_replacement = graph_data.trailing_transpose;
        } else {
            graph_data.last_op_in_sequence_for_replacement = conv_bias;
        }
    } else {
        // TODO: should we want to support also Transpose(NHWC->NCHW) = > conv = > MaxPool => Transpose(NCHW->NHWC)
        // we need to remove continue then
        return false;
    }

    //max pooling
    if ((graph_data.max_pool = DetectNextLayer<opset1::MaxPool>(graph_data.last_op_in_sequence_for_replacement))) {
        if (!VerifyMaxPool(graph_data.max_pool, pool_data))
            return false;

        // disable_nhwc_to_nchw option case
        if (graph_data.trailing_transpose) {
            graph_data.last_op_in_sequence_for_replacement = graph_data.max_pool;
            graph_data.disable_nhwc_to_nchw_option = true;
        } else {
            graph_data.trailing_transpose = DetectNextLayer<Transpose>(graph_data.max_pool);
            graph_data.last_op_in_sequence_for_replacement = graph_data.trailing_transpose;
        }
    }

    //and finally activation function
    if ((graph_data.af = DetectNextLayer<op::util::UnaryElementwiseArithmetic>(graph_data.last_op_in_sequence_for_replacement))) {
        if (!VerifyLayer(graph_data.af))
            return false;

        if (graph_data.trailing_transpose) {
            graph_data.last_op_in_sequence_for_replacement = graph_data.af;
            graph_data.disable_nhwc_to_nchw_option = true;
        } else {
            graph_data.trailing_transpose = DetectNextLayer<Transpose>(graph_data.af);
            graph_data.last_op_in_sequence_for_replacement = graph_data.trailing_transpose;
        }
    }

    if (!graph_data.trailing_transpose || !graph_data.last_op_in_sequence_for_replacement ||
        !TransposeOrderMatches(graph_data.trailing_transpose, { 0, 2, 3, 1 }))
        return false;
    return true;
}

bool CountPadding(const GraphData& graph_data, ConvData& conv_data, const MaxPoolData& pool_data, OutData& out_data) {
    size_t output_channel_count = conv_data.filter_count;

    switch (conv_data.padding_type) {
    case op::PadType::EXPLICIT:
        // all paddings already set
        break;
    case op::PadType::VALID:
        conv_data.pads_begin_y = 0;
        conv_data.pads_begin_x = 0;
        conv_data.pads_end_y = 0;
        conv_data.pads_end_x = 0;
        // all padding equal to 0 - already set
        break;
    case op::PadType::SAME_LOWER:
    case op::PadType::SAME_UPPER:
    {
        out_data.output_height = conv_data.output_shape[2];
        out_data.output_width = conv_data.output_shape[3];

        size_t pad_begin_n_end_y = out_data.output_height * conv_data.filter_stride_y +
            conv_data.filter_height * conv_data.filter_dilation_y - conv_data.input_height - 1;
        size_t pad_begin_n_end_x = out_data.output_width * conv_data.filter_stride_x +
            conv_data.filter_width * conv_data.filter_dilation_x - conv_data.input_width - 1;
        conv_data.pads_begin_y = (op::PadType::SAME_LOWER == conv_data.padding_type) ?
            (pad_begin_n_end_y >> 1) + (pad_begin_n_end_y & 1) : (pad_begin_n_end_y >> 1);
        conv_data.pads_end_y = (op::PadType::SAME_UPPER == conv_data.padding_type) ?
            (pad_begin_n_end_y >> 1) + (pad_begin_n_end_y & 1) : (pad_begin_n_end_y >> 1);
        conv_data.pads_begin_x = (op::PadType::SAME_LOWER == conv_data.padding_type) ?
            (pad_begin_n_end_x >> 1) + (pad_begin_n_end_x & 1) : (pad_begin_n_end_x >> 1);
        conv_data.pads_end_x = (op::PadType::SAME_UPPER == conv_data.padding_type) ?
            (pad_begin_n_end_x >> 1) + (pad_begin_n_end_x & 1) : (pad_begin_n_end_x >> 1);

        break;
    }
    default:
        break;
    }

    out_data.output_height = (conv_data.input_height + conv_data.pads_begin_y + conv_data.pads_end_y -
        ((conv_data.filter_height - 1) * conv_data.filter_dilation_y + 1)) / conv_data.filter_stride_y + 1;
    out_data.output_width = (conv_data.input_width + conv_data.pads_begin_x + conv_data.pads_end_x -
        ((conv_data.filter_width - 1) * conv_data.filter_dilation_x + 1)) / conv_data.filter_stride_x + 1;

    if (output_channel_count != conv_data.output_shape[1] ||
        out_data.output_height != conv_data.output_shape[2] ||
        out_data.output_width != conv_data.output_shape[3]) {
        return false;
    }

    out_data.conv_count = 1;
    // Last check GNA limitations of 768 filters
    size_t total_factorized_conv_channel_count = (conv_data.input_channel_count * conv_data.filter_height * conv_data.filter_width);
    while (total_factorized_conv_channel_count / out_data.conv_count > GNA_MAX_1D_CONV_CHANNEL_COUNT ||
        total_factorized_conv_channel_count % out_data.conv_count != 0)
        out_data.conv_count++;
    //LIMITATION: currently we are able to split only convolutions without pooling in horizontal dimention
    if (out_data.conv_count > GNA_MAX_PERMUTE_COL_COUNT || ((pool_data.pool_size_x > 1 || pool_data.pool_stride_x > 1) && out_data.conv_count > 1))
        return false;

    // GNA supported features - there is no need to decompose such convolution
    if (out_data.conv_count == 1 && (conv_data.input_height == 1 || conv_data.input_width == 1) &&
        conv_data.filter_dilation_x == 1 && conv_data.filter_dilation_y == 1 && !graph_data.disable_nhwc_to_nchw_option)
        return false;

    return true;
}

void ApplyPadding(const GraphData& graph_data, const ConvData& conv_data, OutData& out_data) {
    size_t flat_left_padding = conv_data.input_channel_count * conv_data.pads_begin_x;
    size_t flat_right_padding = conv_data.input_channel_count * conv_data.pads_end_x;
    size_t flat_top_padding = conv_data.input_channel_count * (conv_data.pads_begin_x + conv_data.input_width + conv_data.pads_end_x) * conv_data.pads_begin_y;
    size_t flat_bottom_padding = conv_data.input_channel_count * (conv_data.pads_begin_x + conv_data.input_width + conv_data.pads_end_x) * conv_data.pads_end_y;
    size_t biggest_padding = std::max(std::max(flat_left_padding, flat_right_padding), std::max(flat_top_padding, flat_bottom_padding));
    size_t padded_row_size = conv_data.input_channel_count * (conv_data.pads_begin_x + conv_data.input_width + conv_data.pads_end_x);

    if (conv_data.input_height > 1 && (flat_top_padding > 1 || flat_bottom_padding > 1)) {
        biggest_padding = biggest_padding > padded_row_size ? biggest_padding : padded_row_size;
    }

    auto flat_input = builder::opset1::reshape(graph_data.leading_transpose->input_value(0),
        Shape{ (size_t)1, shape_size(graph_data.leading_transpose->input_value(0).get_shape()) });
    // zero padding
    // TODO: find biggest padding in whole network
    auto const_holding_padding = std::make_shared<opset1::Constant>(element::Type_t::f32, Shape{ 1, biggest_padding }, 0);

    copy_runtime_info(graph_data.conv, const_holding_padding);

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
    out_data.padded_input_plane = flat_input;
    // padding
    if (conv_data.pads_begin_x || conv_data.pads_end_x || conv_data.pads_begin_y || conv_data.pads_end_y) {
        for (size_t p = 0; p < conv_data.pads_begin_y; p++) {
            if (padded_row_size == biggest_padding) {
                input_rows_to_concat.push_back(const_holding_padding);
            } else {
                auto slice = FlatCrop(const_holding_padding, 0, padded_row_size);
                copy_runtime_info(graph_data.conv, slice);
                input_rows_to_concat.push_back(slice);
            }
        }

        // pad every row of input plain
        for (size_t h = 0; h < conv_data.input_height; h++) {
            // left padding     input     right padding
            //     |              |           |
            //     +--------------+-----------+
            //                    |
            //                 concat

            auto not_padded_row = conv_data.input_height == 1 ? flat_input :
                FlatCrop(flat_input, h * conv_data.input_width * conv_data.input_channel_count, conv_data.input_width * conv_data.input_channel_count);
            copy_runtime_info(graph_data.conv, not_padded_row);
            if (flat_left_padding || flat_right_padding) {
                OutputVector single_row_concat_inputs;
                if (flat_left_padding) {
                    if (flat_left_padding == biggest_padding) {
                        single_row_concat_inputs.push_back(const_holding_padding);
                    } else {
                        auto slice = FlatCrop(const_holding_padding, 0, flat_left_padding);
                        copy_runtime_info(graph_data.conv, slice);
                        single_row_concat_inputs.push_back(slice);
                    }
                }
                single_row_concat_inputs.push_back(not_padded_row);
                if (flat_right_padding) {
                    if (flat_right_padding == biggest_padding) {
                        single_row_concat_inputs.push_back(const_holding_padding);
                    } else {
                        auto slice = FlatCrop(const_holding_padding, 0, flat_right_padding);
                        copy_runtime_info(graph_data.conv, slice);
                        single_row_concat_inputs.push_back(slice);
                    }
                }
                auto padded_row_concat = std::make_shared<opset1::Concat>(single_row_concat_inputs, 1);
                copy_runtime_info(graph_data.conv, padded_row_concat);
                input_rows_to_concat.push_back(padded_row_concat);
            } else {
                input_rows_to_concat.push_back(not_padded_row);
            }
        }

        // Bottom padding
        for (size_t p = 0; p < conv_data.pads_end_y; p++) {
            if (padded_row_size == biggest_padding) {
                input_rows_to_concat.push_back(const_holding_padding);
            } else {
                auto slice = FlatCrop(const_holding_padding, 0, padded_row_size);
                copy_runtime_info(graph_data.conv, slice);
                input_rows_to_concat.push_back(slice);
            }
        }
        out_data.padded_input_plane = std::make_shared<opset1::Concat>(input_rows_to_concat, 1);
        copy_runtime_info(graph_data.conv, out_data.padded_input_plane);
    }
}

std::shared_ptr<Node> GetPartialResults(const GraphData& graph_data, ConvData& conv_data, const OutData& out_data, const MaxPoolData& pool_data,
    Output<Node>& reduced_input_plane, const std::vector<std::shared_ptr<opset1::Constant>>& h_1_filters, const size_t conv_index) {
    OutputVector result_chunks;
    std::shared_ptr<Node> last_op;
    bool horizontal_permute = (conv_data.filter_dilation_x > 1);
    size_t h_1_filter_channel_count = (conv_data.input_channel_count * conv_data.filter_height);

    for (size_t y = 0; y < out_data.output_height; y += conv_data.filter_stride_y) {
        size_t offset = y * (conv_data.pads_begin_x + conv_data.input_width + conv_data.pads_end_x) * h_1_filter_channel_count;
        auto row = (out_data.output_height == 1) ? reduced_input_plane :
            FlatCrop(reduced_input_plane, offset, (conv_data.pads_begin_x + conv_data.input_width + conv_data.pads_end_x) * h_1_filter_channel_count);
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
            for (size_t f_x = 0; f_x < conv_data.filter_width; f_x++) {
                size_t offset = f_x * conv_data.filter_dilation_x * h_1_filter_channel_count;
                // point wise convolutions - as many as output width
                auto slice = FlatCrop(row, offset, h_1_filter_channel_count * out_data.output_width);
                copy_runtime_info(graph_data.conv, slice);
                dilated_chunks.push_back(slice);
            }
            // concat
            auto dilated_chunks_concat = std::make_shared<opset1::Concat>(dilated_chunks, 0);

            // permute
            auto permuted_dilated_chunks = builder::opset1::transpose(dilated_chunks_concat);

            // flatten
            auto flatten_dilated_conv_input = builder::opset1::reshape(permuted_dilated_chunks,
                Shape{ (size_t)1, 1, out_data.output_width, h_1_filter_channel_count * conv_data.filter_width });
            copy_runtime_info(graph_data.conv, { flatten_dilated_conv_input, permuted_dilated_chunks, dilated_chunks_concat });

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
                    last_conv_block_op = std::make_shared<opset1::MaxPool>(last_conv_block_op, Strides{ 1, pool_stride_x },
                        Shape{ 0, 0 }, Shape{ 0, 0 }, Shape{ 1, pool_size_x }, rounding_type, op::PadType::VALID);
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
            size_t padded_row_width = conv_data.pads_begin_x + conv_data.input_width + conv_data.pads_end_x;
            size_t padded_row_flat_width = shape_size(nhwc_conv_y_input.get_shape());
            nhwc_conv_y_input = builder::opset1::reshape(nhwc_conv_y_input, { 1ull, 1ull, padded_row_width, padded_row_flat_width / padded_row_width });
        }

        // valid 1D convolution wrapped with permutes NHWC => NCHW => conv => NCHW => NHWC
        // activation function can be fused with convolution only if it is not split
        auto nhwc_y_output = nhwc_conv_1d(graph_data.conv, nhwc_conv_y_input, h_1_filters[conv_index], conv_index ? nullptr : graph_data.bias_const,
            conv_data.filter_stride_x, pool_data.pool_size_x, pool_data.pool_stride_x,
            graph_data.max_pool ? graph_data.max_pool->get_rounding_type() : RoundingType::FLOOR, out_data.conv_count == 1 ? graph_data.af : nullptr, y);
        result_chunks.push_back(nhwc_y_output);
        last_op = nhwc_y_output;
    }

    // Vertical dimemsion greater than 1
    if (result_chunks.size() > 1) {
        // concat in H dim
        // in NHWC index of H is 1
        auto concatenated_sub_results = std::make_shared<opset1::Concat>(result_chunks, 1);
        copy_runtime_info(graph_data.conv, concatenated_sub_results);
        last_op = concatenated_sub_results;
    }
    return last_op;
}

void ApplyTransform(const GraphData& graph_data, ConvData& conv_data, const MaxPoolData& pool_data, const OutData& out_data) {
    const Output<Node>& filters = graph_data.conv->input_value(1);
    auto filter_values = std::dynamic_pointer_cast<opset1::Constant>(filters.get_node_shared_ptr());
    OutputVector split_planes;
    if (out_data.conv_count > 1) {
        auto reshape_before_permute = builder::opset1::reshape(out_data.padded_input_plane,
            Shape{ shape_size(out_data.padded_input_plane->get_shape()) / out_data.conv_count, out_data.conv_count });
        auto permute_before_channel_wise_split = builder::opset1::reorder_axes(reshape_before_permute, { 1ull, 0ull });
        auto reshape_after_permute = builder::opset1::reshape(permute_before_channel_wise_split,
            Shape{ (size_t)out_data.conv_count, out_data.padded_input_plane->get_shape()[1] / out_data.conv_count });
        split_planes = builder::opset1::split(reshape_after_permute, out_data.conv_count, 0);
    } else {
        split_planes.push_back(out_data.padded_input_plane);
    }

    bool vertical_permute = (conv_data.filter_height > 1);
    bool horizontal_permute = (conv_data.filter_dilation_x > 1);
    std::vector<std::shared_ptr<opset1::Constant>> h_1_filters =
        ReduceConv2DFilterHeightByChannelPermute(filter_values, vertical_permute, horizontal_permute, out_data.conv_count);
    for (auto filter : h_1_filters)
        copy_runtime_info(graph_data.conv, filter);

    // if we split input planes due to GNA limitation - we must sum their results
    std::vector<std::shared_ptr<Node>> partial_conv_results;
    conv_data.input_channel_count /= out_data.conv_count;

    for (size_t conv_index = 0; conv_index < out_data.conv_count; conv_index++) {
        Output<Node> reduced_input_plane = split_planes[conv_index];
        // lets change filter height to 1
        if (conv_data.filter_height > 1) {
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
            for (size_t f_y = 0; f_y < conv_data.filter_height; f_y++) {
                size_t offset = f_y * conv_data.filter_dilation_y * (conv_data.pads_begin_x + conv_data.input_width + conv_data.pads_end_x) *
                    conv_data.input_channel_count;
                // point wise convolutions - as many as output width
                auto slice = FlatCrop(reduced_input_plane, offset, (conv_data.pads_begin_x + conv_data.input_width + conv_data.pads_end_x) *
                    conv_data.input_channel_count * out_data.output_height);
                copy_runtime_info(graph_data.conv, slice);
                dilated_input_planes.push_back(slice);
            }
            // now let's flatten kernel of convolution in vertical dimension
            // it is done by interleaving dilated input planes
            auto dilated_chunks_concat = std::make_shared<opset1::Concat>(dilated_input_planes, 0);

            auto permuted_dilated_chunks = builder::opset1::transpose(dilated_chunks_concat);
            // flatten
            auto flatten_dilated_permuted_input = builder::opset1::reshape(permuted_dilated_chunks,
                Shape{ (size_t)1, (conv_data.pads_begin_x + conv_data.input_width + conv_data.pads_end_x) *
                conv_data.input_channel_count * out_data.output_height * conv_data.filter_height });

            copy_runtime_info(graph_data.conv, { dilated_chunks_concat, flatten_dilated_permuted_input, permuted_dilated_chunks });
            reduced_input_plane = flatten_dilated_permuted_input;
        }

        partial_conv_results.push_back(GetPartialResults(graph_data, conv_data, out_data, pool_data, reduced_input_plane, h_1_filters, conv_index));
    }

    std::shared_ptr<Node> conv_result = partial_conv_results[0];
    for (size_t i = 1; i < partial_conv_results.size(); i++) {
        auto add_result = std::make_shared<opset1::Add>(partial_conv_results[i], conv_result);
        copy_runtime_info(graph_data.conv, add_result);
        conv_result = add_result;
    }

    //TODO: maxpool 2d case
    //if (graph_data.max_pool && (maxpool_data.pool_size_y > 1 || maxpool_data.pool_stride_y > 1)) {
    //}

    // activation function
    if (graph_data.af && out_data.conv_count > 1) {
        auto af_result = graph_data.af->copy_with_new_inputs({ conv_result });
        copy_runtime_info(graph_data.conv, af_result);
        conv_result = af_result;
    }
    // we need to put the same name as before for the conv layer, so its output can be used as network result
    std::string conv_result_name = graph_data.last_op_in_sequence_for_replacement->get_friendly_name();
    replace_node(graph_data.last_op_in_sequence_for_replacement, conv_result);
    conv_result->set_friendly_name(conv_result_name);
}
} // namespace

// Supported cases :
// - Transpose(NHWC->NCHW) = > conv = > Transpose(NCHW->NHWC)
// - Transpose(NHWC->NCHW) = > conv = > broadcasted add(BIAS) = > Transpose(NCHW->NHWC)
// - Transpose(NHWC->NCHW) = > conv = > broadcasted add(BIAS) = > MaxPooling = > Transpose(NCHW->NHWC) (TODO: 2d max pool case)
// - Transpose(NHWC->NCHW) = > conv = > broadcasted add(BIAS) = > ActivationFunction = > Transpose(NCHW->NHWC)
// - Transpose(NHWC->NCHW) = > conv = > broadcasted add(BIAS) = > MaxPool = > ActivationFunction = > Transpose(NCHW->NHWC) (TODO: 2d max pool case)
// - Transpose(NHWC->NCHW) = > conv = > Transpose(NCHW->NHWC) = > BIAS(output of MO --disable_nhwc_to_nchw option)
// - Transpose(NHWC->NCHW) = > conv = > Transpose(NCHW->NHWC) = > BIAS = > AF(output of MO --disable_nhwc_to_nchw option)

NGRAPH_RTTI_DEFINITION(pass::Conv2dDecomposition, "Conv2dDecomposition", 0);
bool pass::Conv2dDecomposition::run_on_function(std::shared_ptr<Function> f) {
    // Traverse nGraph Function in topological order
    bool is_graph_modfied = false;

    for (auto& node : f->get_ordered_ops()) {
        GraphData graph_data;
        ConvData conv_data;
        MaxPoolData maxpool_data { 1, 1 };
        OutData out_data { 0, 0 };

        if ((graph_data.conv = DetectVerifyConvolution(node)) == nullptr)
            continue;

        transformation_callback(graph_data.conv);

        FillConvData(graph_data.conv, conv_data);

        // TODO: Check if filter weights are not dynamic
        // TODO: Check BIAS sizes

        // we are looking for Transpose(NHWC->NCHW) => conv => Transpose(NCHW->NHWC)
        // or similar cases so required network must be in NHWC order like in TF
        if (!(graph_data.leading_transpose = DetectVerifyLeadingTranspose(graph_data.conv)))
            continue;

        if (!DetectGraphSequence(graph_data, conv_data, maxpool_data))
            continue;

        if (!CountPadding(graph_data, conv_data, maxpool_data, out_data))
            continue;

        // All checks applied - now we may start to do transformations
        ApplyPadding(graph_data, conv_data, out_data);
        ApplyTransform(graph_data, conv_data, maxpool_data, out_data);

        is_graph_modfied = true;
    }
    return is_graph_modfied;
}
