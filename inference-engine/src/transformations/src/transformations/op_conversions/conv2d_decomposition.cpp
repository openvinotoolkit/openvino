// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/conv2d_decomposition.hpp"

#include <memory>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/builder/reshape.hpp>
#include <ngraph/builder/split.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include "itt.hpp"

using namespace ngraph;
using namespace op;

//#define RESHAPE_NCHW_NHWC

std::vector<std::shared_ptr<opset1::Constant>> Convert2DFilter2PointwiseFilterSet(std::shared_ptr<opset1::Constant> & filters)
{
    std::vector <std::shared_ptr<opset1::Constant>> result;
    auto filter_shape = filters->get_output_shape(0);
    if (filter_shape.size() == 4)
    {
        std::vector<std::vector<float>> pointwise_filters;
        pointwise_filters.resize(filter_shape[2] * filter_shape[3]);
        for (size_t i = 0; i < filter_shape[2] * filter_shape[3]; i++) {
            pointwise_filters[i].resize(filter_shape[0] * filter_shape[1]);
        }
        size_t offset = 0;
        const float* data = filters->get_data_ptr<float>();
        for (size_t n = 0; n < filter_shape[0]; n++) {
            for (size_t c = 0; c < filter_shape[1]; c++) {
                for (size_t h = 0; h < filter_shape[2]; h++) {
                    for (size_t w = 0; w < filter_shape[3]; w++) {
                        pointwise_filters[h * filter_shape[3] + w][n * filter_shape[1] + c] = data[offset++];
                    }
                }
            }
        }
        for (auto pwf : pointwise_filters) {
            result.push_back(std::make_shared<opset1::Constant>(element::f32, Shape{ filter_shape[0], filter_shape[1]* filter_shape[1]* filter_shape[1], 1, 1}, pwf));
        }
    }

    return result;
}

std::shared_ptr<opset1::Constant> ReduceConv2DFilterHeightByChannelPermute(std::shared_ptr<opset1::Constant>& filters)
{
    std::shared_ptr<opset1::Constant> result;
    auto filter_shape = filters->get_output_shape(0);

    if (filter_shape.size() == 4)
    {
        std::vector<float> flat_filters;
        flat_filters.resize(shape_size(filter_shape));

        size_t offset = 0;
        auto N = filter_shape[0];
        auto C = filter_shape[1];
        auto H = filter_shape[2];
        auto W = filter_shape[3];
        const float* data = filters->get_data_ptr<float>();
        for (size_t n = 0; n < N; n++) {
            for (size_t c = 0; c < C; c++) {
                for (size_t h = 0; h < filter_shape[2]; h++) {
                    for (size_t w = 0; w < filter_shape[3]; w++) {
                        // NCHW=>NC'1W
                        //flat_filters[n * C * H * W +
                        //    (c * H * W) + w * H + h] = data[offset];
                        flat_filters[offset] = data[offset];
                        offset++;
                    }
                }
            }
        }

        result = std::make_shared<opset1::Constant>(element::f32, Shape{ filter_shape[0], filter_shape[1] * filter_shape[2] * filter_shape[3], 1, 1 }, flat_filters);
    }

    return result;
}

std::shared_ptr<opset1::StridedSlice> FlatCrop(Output<Node> input, size_t offset, size_t size)
{
    auto shape = input.get_shape();
    if (shape.size() == 1) {
        return std::make_shared<ngraph::opset1::StridedSlice>(
            input, // data
            ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{ 1 }, { offset }), // begin slice index
            ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{ 1 }, { offset + size }), // end slice index
            ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{ 1 }, { 1 }), // strides
            std::vector<int64_t>{0}, // begin mask
            std::vector<int64_t>{0} // end mask
        );
    }
    else if (shape.size() == 2) {
        return std::make_shared<ngraph::opset1::StridedSlice>(
            input, // data
            ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{ 2 }, { (size_t)0, offset }), // begin sice index
            ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{ 2 }, { (size_t)0, offset + size }), // end slice index
            ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{ 2 }, { (size_t)1, (size_t)1 }), // strides
            std::vector<int64_t>{1, 0}, // begin mask
            std::vector<int64_t>{1, 0} // end mask
        );
    }
    return nullptr;
}

NGRAPH_RTTI_DEFINITION(ngraph::pass::Conv2dDecomposition, "Conv2dDecomposition", 0);
bool ngraph::pass::Conv2dDecomposition::run_on_function(std::shared_ptr<ngraph::Function> f) {
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

        if (input.get_shape().size() != 4 ||
            filters.get_shape().size() != 4 ||
            output_shape.size() != 4 ||
            conv->get_dilations().size() != 2 ||
            conv->get_strides().size() != 2 ||
            input.get_shape()[0] != 1) {
            // we support only 2D conv batch 1
            continue;
        }
        // we are looking for NHWC Transpose NCHW => conv => NCHW Transpose NHWC
        auto leading_transpose = std::dynamic_pointer_cast<Transpose>(input.get_node_shared_ptr());
        auto output_0 = node->get_output_target_inputs(0);
        if (output_0.size() != 1)
            continue;
        // TODO: check if next is NHWC => NCHW
        auto output_0_node = output_0.begin()->get_node()->shared_from_this();

        auto trailing_transpose = std::dynamic_pointer_cast<Transpose>(output_0_node);
        if (!trailing_transpose)
            continue;
        // TODO: check if next is NCHW => NHWC
        auto filter_values = std::dynamic_pointer_cast<ngraph::opset1::Constant>(filters.get_node_shared_ptr());
        if (!filter_values) {
            continue;
        }
        size_t input_channel_count = input.get_shape()[1];
        size_t input_height = input.get_shape()[2];
        size_t input_width = input.get_shape()[3];

        if (input_channel_count < 1 ||
            0 != input_channel_count % 32) {
            // we do only support conv2d with channels k * 32
            continue;
        }

        size_t filter_count = filters.get_shape()[0];
        size_t filter_channel_count = filters.get_shape()[1];
        size_t filter_height = filters.get_shape()[2];
        size_t filter_width = filters.get_shape()[3];

        size_t filter_dilation_x = conv->get_dilations()[1];
        size_t filter_dilation_y = conv->get_dilations()[0];

        size_t filter_stride_x = conv->get_strides()[1];
        size_t filter_stride_y = conv->get_strides()[0];

        // we are assuming VALID conv
        size_t pads_begin_x = 0;
        size_t pads_begin_y = 0;
        size_t pads_end_x = 0;
        size_t pads_end_y = 0;

        size_t output_channel_count = filter_channel_count;
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

            size_t pad_begin_n_end_y = output_height * filter_stride_y + filter_height * filter_dilation_y - input_height;
            size_t pad_begin_n_end_x = output_width * filter_stride_x + filter_width * filter_dilation_x - input_width;
            pads_begin_y = (ngraph::op::PadType::SAME_LOWER == padding_type) ? (pad_begin_n_end_y >> 1) + (pad_begin_n_end_y & 1) : (pad_begin_n_end_y >> 1);
            pads_end_y = (ngraph::op::PadType::SAME_UPPER == padding_type) ? (pad_begin_n_end_y >> 1) + (pad_begin_n_end_y & 1) : (pad_begin_n_end_y >> 1);
            pads_begin_x = (ngraph::op::PadType::SAME_LOWER == padding_type) ? (pad_begin_n_end_y >> 1) + (pad_begin_n_end_y & 1) : (pad_begin_n_end_y >> 1);
            pads_end_x = (ngraph::op::PadType::SAME_UPPER == padding_type) ? (pad_begin_n_end_y >> 1) + (pad_begin_n_end_y & 1) : (pad_begin_n_end_y >> 1);

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

        size_t flat_left_padding = input_channel_count * pads_begin_x;
        size_t flat_right_padding = input_channel_count * pads_end_x;
        size_t flat_top_padding = input_channel_count * (pads_begin_x + input_width + pads_end_x) * pads_begin_y;
        size_t flat_bottom_padding = input_channel_count * (pads_begin_x + input_width + pads_end_x) * pads_end_y;
        size_t biggest_padding = std::max(std::max(flat_left_padding, flat_right_padding), std::max(flat_top_padding, flat_bottom_padding));
        size_t padded_row_size = input_channel_count * (pads_begin_x + input_width + pads_end_x);

        if (input_height > 1 && (flat_top_padding > 1 || flat_bottom_padding > 1)) {
            biggest_padding = biggest_padding > padded_row_size ? biggest_padding : padded_row_size;
        }



        auto flat_input = builder::opset1::reshape(leading_transpose->input_value(0), Shape{ (size_t)1, shape_size(leading_transpose->input_value(0).get_shape()) });
        // we start injecting subgraph
        ngraph::copy_runtime_info(leading_transpose, flat_input);
        ngraph::replace_node(leading_transpose, flat_input);

        // zero padding
        // TODO: find biggest padding in whole network
        auto const_holding_padding = std::make_shared<opset1::Constant>(element::Type_t::f32, Shape{ 1, biggest_padding }, 0);

        ngraph::copy_runtime_info(conv, const_holding_padding );

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
        for (int h = 0; h < input_height; h++) {

            // left padding     input     right padding
            //     |              |           |
            //     +--------------+-----------+
            //                    |
            //                 concat

            auto not_padded_row = FlatCrop(flat_input, h * input_width * input_channel_count, input_width * input_channel_count);
            ngraph::copy_runtime_info(conv, not_padded_row);
            if (flat_left_padding || flat_right_padding) {
                OutputVector single_row_concat_inputs;
                if (flat_left_padding) {
                    if (flat_left_padding == biggest_padding) {
                        single_row_concat_inputs.push_back(const_holding_padding);
                    }
                    else {
                        auto slice = FlatCrop(const_holding_padding, 0, flat_left_padding);
                        ngraph::copy_runtime_info(conv, slice);
                        single_row_concat_inputs.push_back(slice);
                    }
                }
                single_row_concat_inputs.push_back(not_padded_row);
                if (flat_right_padding) {
                    if (flat_right_padding == biggest_padding) {
                        single_row_concat_inputs.push_back(const_holding_padding);
                    }
                    else {
                        auto slice = FlatCrop(const_holding_padding, 0, flat_right_padding);
                        ngraph::copy_runtime_info(conv, slice);
                        single_row_concat_inputs.push_back(slice);
                    }
                }
                auto padded_row_concat = std::make_shared<opset1::Concat>(single_row_concat_inputs, 0);
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

        auto padded_input_plane = flat_input;
        if (pads_begin_x || pads_end_y || filter_height > 1) {
            padded_input_plane = std::make_shared<opset1::Concat>(input_rows_to_concat, 1);
            ngraph::copy_runtime_info(conv, padded_input_plane);
        }

        // lets change filter height to 1
        if (filter_height > 1) {
            //                padded row - NHWC order
            //                    |
            //          split in vertical dim ( filter height)
            //                  / | \
            //                  concat
            //                    |
            //                 permute

            OutputVector dilated_input_planes;
            for (size_t f_y = 0; f_y < filter_height; f_y++) {
                size_t offset = f_y * filter_dilation_y * (pads_begin_x + input_width + pads_end_x) * input_channel_count;
                // point wise convolutions - as many as output width
                auto slice = FlatCrop(padded_input_plane, offset, (pads_begin_x + input_width + pads_end_x) * input_channel_count * output_height);
                ngraph::copy_runtime_info(conv, slice);
                dilated_input_planes.push_back(slice);
            }
            // now lets flatten kernel of convolution in vertical dimenson
            // it is done by interleaving dilated input planes
            auto dilated_chunks_concat = std::make_shared<opset1::Concat>(dilated_input_planes, 0);

            auto permuted_dilated_chunks = builder::opset1::transpose(dilated_chunks_concat);
            // flatten
            auto flatten_dilated_permuted_input = builder::opset1::reshape(permuted_dilated_chunks,
                Shape{ (size_t)1, (pads_begin_x + input_width + pads_end_x) * input_channel_count * output_height * filter_height });

            ngraph::copy_runtime_info(conv, { dilated_chunks_concat,flatten_dilated_permuted_input, permuted_dilated_chunks});
            padded_input_plane = flatten_dilated_permuted_input;
        }
        OutputVector result_chunks;
        std::shared_ptr<Node> last_op;
        size_t h_1_filter_channel_count = (input_channel_count * filter_height);

        bool vertical_permute = (filter_height > 1);
        bool horizontal_permute = (filter_dilation_x > 1);

        auto h_1_filters = vertical_permute ? ReduceConv2DFilterHeightByChannelPermute(filter_values) : filter_values;
        if (vertical_permute)
        {
            ngraph::copy_runtime_info(conv, h_1_filters);
        }
        for (size_t y = 0; y < output_height; y+= filter_stride_y) {
            size_t offset = y * (pads_begin_x + input_width + pads_end_x) * h_1_filter_channel_count;
            auto row = (output_height == 1) ? padded_input_plane :
                FlatCrop(padded_input_plane, offset, (pads_begin_x + input_width + pads_end_x) * h_1_filter_channel_count);
            //                padded row
            //                    |
            //          ??? <dilation !=1> ???
            //                    |
            //          split in vertical dim
            //                  / | \
            //                  concat
            //                    |
            //                 permute
            //                    |
            //             permute NHWC => NCHW
            //                    |
            //                 conv 1D
            //                    |
            //             permute NCHW => NHWC

            //TODO: handle limitation of GNA permute
            auto nhwc_conv_y_input = row;
            if (horizontal_permute) {
                // split
                OutputVector dilated_chunks;
                for (size_t f_x = 0; f_x < filter_width; f_x++) {
                    size_t offset = f_x * filter_dilation_x * h_1_filter_channel_count;
                    // point wise convolutions - as many as output width
                    auto slice = FlatCrop(row, offset, h_1_filter_channel_count * output_width);
                    ngraph::copy_runtime_info(conv, slice);
                    dilated_chunks.push_back(slice);
                }
                // concat
                auto dilated_chunks_concat = std::make_shared<opset1::Concat>(dilated_chunks, 0);

                // permute
                auto permuted_dilated_chunks = builder::opset1::transpose(dilated_chunks_concat);

                // flatten
                auto flatten_dilated_conv_input = builder::opset1::reshape(permuted_dilated_chunks, Shape{ (size_t)1, 1, output_width, h_1_filter_channel_count * filter_width});
                ngraph::copy_runtime_info(conv, { flatten_dilated_conv_input, permuted_dilated_chunks, dilated_chunks_concat });

                nhwc_conv_y_input = flatten_dilated_conv_input;
            }

            // valid 1D convolution wrapped with permutes NHWC => NCHW => conv => NCHW => NHWC
            // NHWC => NCHW
            auto nchw_y_input = builder::opset1::reorder_axes(nhwc_conv_y_input, { 0ull,3ull,1ull,2ull });
            // conv
            auto conv_y = std::make_shared<opset1::Convolution>(nchw_y_input, h_1_filters,
                Strides{ 1, filter_stride_x }, CoordinateDiff{ 0, 0 }, CoordinateDiff{ 0, 0 }, Strides{ 1, 1 }, PadType::VALID);
            std::string conv_y_name = conv->get_friendly_name() + "_H_" + std::to_string(y);
            conv_y->set_friendly_name(conv_y_name);
            // NCHW => NHWC
            auto nhwc_y_output = builder::opset1::reorder_axes(conv_y, { 0ull,2ull,3ull,1ull });


            ngraph::copy_runtime_info(conv, { nchw_y_input, conv_y, nhwc_y_output });
            result_chunks.push_back(nhwc_y_output);
            last_op = nhwc_y_output;
        }
        if (result_chunks.size() > 1) {
            // concat in H dim
            // in NHWC index of H is 1
            auto concatenated_sub_results = std::make_shared<opset1::Concat>(result_chunks, 1);
            ngraph::copy_runtime_info(conv, concatenated_sub_results);
            last_op = concatenated_sub_results;
        }

        // we need to put friendly name, so the conv output can be used as network result
        std::string last_op_name = trailing_transpose->get_friendly_name();

        ngraph::replace_node(trailing_transpose, last_op);
        last_op->set_friendly_name(last_op_name);

        is_graph_modfied = true;
    }
    return is_graph_modfied;
}

//auto pointwise_filters = Convert2DFilter2PointwiseFilterSet(filter_values);
//if (pointwise_filters.size() == filter_width) {
//    auto pointwise_conv_input = FlatCrop(padded_input_plane, 0, h_1_filter_channel_count * output_width);
//
//    auto conv_output = std::make_shared<opset1::Convolution>(pointwise_conv_input, pointwise_filters[0],
//        Strides{ filter_stride_y, filter_stride_x }, CoordinateDiff{ 0, 0 }, CoordinateDiff{ 0, 0 }, Strides{ 1, 1 }, PadType::VALID);
//
//    ngraph::copy_runtime_info(conv, { conv_output, pointwise_conv_input });
//
//    for (size_t f_x = 1; f_x < filter_width; f_x++) {
//        size_t offset = f_x * filter_dilation_x * h_1_filter_channel_count;
//        // point wise convolutions - as many as output width
//        auto pointwise_conv_input = FlatCrop(padded_input_plane, offset, h_1_filter_channel_count * output_width);
//
//        auto pointwise_conv_output = std::make_shared<opset1::Convolution>(pointwise_conv_input, pointwise_filters[f_x],
//            Strides{ filter_stride_y, filter_stride_x }, CoordinateDiff{ 0, 0 }, CoordinateDiff{ 0, 0 }, Strides{ 1, 1 }, PadType::VALID);
//
//        conv_output = std::make_shared <opset1::Add>(pointwise_conv_output, conv_output);
//        ngraph::copy_runtime_info(conv, { conv_output, pointwise_conv_input, pointwise_conv_output, });
//    }
//    ngraph::replace_node(conv, conv_output);
//    is_graph_modfied = true;
//}
//else {
//    continue;
//}
