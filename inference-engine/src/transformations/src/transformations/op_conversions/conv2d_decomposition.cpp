// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/conv2d_decomposition.hpp"

#include <memory>

#include <ngraph/opsets/opset1.hpp>

#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include "itt.hpp"

using namespace ngraph;
using namespace op;

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
            result.push_back(std::make_shared<opset1::Constant>(element::f32, Shape{ filter_shape[0], filter_shape[1], 1, 1}, pwf));
        }
    }

    return result;
}

std::shared_ptr<opset1::Constant> Flatten2DFilterByChannelPermute(std::shared_ptr<opset1::Constant>& filters)
{
    std::shared_ptr<opset1::Constant> result;
    auto filter_shape = filters->get_output_shape(0);
    if (filter_shape.size() == 4)
    {
        std::vector<float> flat_filters;
        flat_filters.resize(shape_size(filter_shape));

        size_t offset = 0;
        const float* data = filters->get_data_ptr<float>();
        for (size_t n = 0; n < filter_shape[0]; n++) {
            for (size_t c = 0; c < filter_shape[1]; c++) {
                for (size_t h = 0; h < filter_shape[2]; h++) {
                    for (size_t w = 0; w < filter_shape[3]; w++) {
                        // NCHW=>NC'H1
                        flat_filters[n * filter_shape[1] * filter_shape[2] * filter_shape[3] +
                            (c * filter_shape[3] + w) * filter_shape[2] + h] = data[offset++];
                    }
                }
            }
        }

        result = std::make_shared<opset1::Constant>(element::f32, Shape{ filter_shape[0], filter_shape[1]*filter_shape[3], filter_shape[2], 1 }, flat_filters);
    }

    return result;
}


std::vector<std::shared_ptr<opset1::Constant>> Convert2DFilter2PointwiseFilterSet(std::shared_ptr<opset1::Constant>& filters)
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
            result.push_back(std::make_shared<opset1::Constant>(element::f32, Shape{ filter_shape[0], filter_shape[1], 1, 1 }, pwf));
        }
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
            return false;
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

        size_t filter_dilation_y = conv->get_dilations()[0];
        size_t filter_dilation_x = conv->get_dilations()[1];

        size_t filter_stride_y = conv->get_strides()[0];
        size_t filter_stride_x = conv->get_strides()[1];

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
        output_height = (input_height + pads_begin_y + pads_end_y - filter_height * filter_dilation_y) / filter_stride_y;
        output_width = (input_width + pads_begin_x + pads_end_x - filter_width * filter_dilation_x) / filter_stride_x;

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
        auto flat_input = std::make_shared<opset1::Reshape>(input, Shape{ (size_t)1, shape_size(input.get_shape()) });
        // zero padding
        // TODO: find biggest padding in whole network
        auto const_holding_padding = std::make_shared<opset1::Constant>(element::Type_t::f32, Shape{ 1, biggest_padding }, 0);

        ngraph::copy_runtime_info(conv, { flat_input, const_holding_padding });

        if (input_height == 1) {
            Output<Node>& prev = conv->input_value(0);
            // left padding     input     right padding
            //     |              |           |
            //     +--------------+-----------+
            //                    |
            //                 concat
            //                    |
            //          ??? <dilation !=1> ???
            //                    |
            //                  split
            //                  / | \
            //                  concat
            //                    |
            //                 permute
            //                    |
            //                 conv 1D

            if (flat_left_padding || flat_right_padding) {
                OutputVector concat_inputs;
                if (flat_left_padding) {
                    if (flat_left_padding == biggest_padding) {
                        concat_inputs.push_back(const_holding_padding);
                    } else {
                        auto slice = FlatCrop(const_holding_padding, 0, flat_left_padding);
                        ngraph::copy_runtime_info(conv, slice);
                        concat_inputs.push_back(slice);
                    }
                }
                concat_inputs.push_back(input);
                if (flat_right_padding) {
                    if (flat_right_padding == biggest_padding) {
                        concat_inputs.push_back(const_holding_padding);
                    } else {
                        auto slice = FlatCrop(const_holding_padding, 0, flat_right_padding);
                        ngraph::copy_runtime_info(conv, slice);
                        concat_inputs.push_back(slice);
                    }
                }
                auto padded_row_concat = std::make_shared<opset1::Concat>(concat_inputs, 0);
                ngraph::copy_runtime_info(conv, padded_row_concat);
                prev = padded_row_concat;
            }
            // limitation of GNA permute
            if (filter_width <= 8 && 0 == (output_width % 8)) {
                // split
                OutputVector concat_inputs;
                for (size_t f_x = 0; f_x < filter_width; f_x++) {
                    size_t offset = f_x * filter_dilation_x * input_channel_count;
                    // point wise convolutions - as many as output width
                    auto slice = FlatCrop(prev, offset, input_channel_count * output_width);
                    ngraph::copy_runtime_info(conv, slice);
                    concat_inputs.push_back(slice);
                }
                // concat
                auto dilated_chunks_concat = std::make_shared<opset1::Concat>(concat_inputs, 1);

                // permute
                auto const_holding_transpose_shape = std::make_shared<opset1::Constant>(element::Type_t::i64, Shape{ 2 }, std::vector<size_t>{0,1});
                auto permuted_dilated_chunks = std::make_shared<opset1::Transpose>(dilated_chunks_concat, const_holding_transpose_shape);

                // flatten
                auto flatten_dilated_input = std::make_shared<opset1::Reshape>(permuted_dilated_chunks, Shape{ (size_t)1, input_channel_count * output_width * filter_width });

                auto permuted_filters = FlattenWidthOf2DFilterByChannelPermute(filter_values);

                // valid convolution
                auto conv_output = std::make_shared<opset1::Convolution>(flatten_dilated_input, permuted_filters,
                    Strides{ filter_stride_y, filter_stride_x }, CoordinateDiff{ 0, 0 }, CoordinateDiff{ 0, 0 }, Strides{ 1, 1 }, PadType::VALID);

                ngraph::copy_runtime_info(conv, { conv_output, conv_output, permuted_filters, flatten_dilated_input, permuted_dilated_chunks, const_holding_transpose_shape, dilated_chunks_concat });

                ngraph::replace_node(conv, conv_output);
                is_graph_modfied = true;
            } else {
                auto pointwise_filters = Convert2DFilter2PointwiseFilterSet(filter_values);
                if (pointwise_filters.size() == filter_width) {
                    auto pointwise_conv_input = FlatCrop(prev, 0, input_channel_count * output_width);
                    
                    auto conv_output = std::make_shared<opset1::Convolution>(pointwise_conv_input, pointwise_filters[0],
                        Strides{ filter_stride_y, filter_stride_x}, CoordinateDiff{ 0, 0 }, CoordinateDiff{ 0, 0 }, Strides{ 1, 1 }, PadType::VALID);

                    ngraph::copy_runtime_info(conv, { conv_output, pointwise_conv_input });

                    for (size_t f_x = 1; f_x < filter_width; f_x++) {
                        size_t offset = f_x * filter_dilation_x * input_channel_count;
                        // point wise convolutions - as many as output width
                        auto pointwise_conv_input = FlatCrop(prev, offset, input_channel_count * output_width);

                        auto pointwise_conv_output = std::make_shared<opset1::Convolution>(pointwise_conv_input, pointwise_filters[f_x],
                            Strides{ filter_stride_y, filter_stride_x }, CoordinateDiff{ 0, 0 }, CoordinateDiff{ 0, 0 }, Strides{ 1, 1 }, PadType::VALID);

                        conv_output = std::make_shared <opset1::Add>(pointwise_conv_output, conv_output);
                        ngraph::copy_runtime_info(conv, { conv_output, pointwise_conv_input, pointwise_conv_output,  });
                    }
                    ngraph::replace_node(conv, conv_output);
                    is_graph_modfied = true;
                } else {
                    continue;
                }
            }
        } else {
            // padding
            // padding
            // ... row ...
            // ... row ...
            // ...........
            // ... row ...
            // padding
            // padding

            // Pad input plane row by row
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
            // rows with activations values
            for (int h = 0; h < input_height; h++) {
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
            // padding
            for (size_t p = 0; p < pads_end_y; p++) {
                if (padded_row_size == biggest_padding) {
                    input_rows_to_concat.push_back(const_holding_padding);
                } else {
                    auto slice = FlatCrop(const_holding_padding, 0, padded_row_size);
                    ngraph::copy_runtime_info(conv, slice);
                    input_rows_to_concat.push_back(slice);
                }
            }
            auto padded_input_plane = std::make_shared<opset1::Concat>(input_rows_to_concat, 0);

            if (filter_height > 1) {
                OutputVector padded_input_plane_filter_height_one;
                OutputVector output_plane;

                for (size_t f_y = 0; f_y < filter_height; f_y++) {
                    size_t offset = f_y * filter_dilation_y * (pads_begin_x + input_width + pads_end_x) * input_channel_count;
                    // point wise convolutions - as many as output width
                    auto slice = FlatCrop(padded_input_plane, offset, (pads_begin_x + input_width + pads_end_x) * input_channel_count * output_height);
                    ngraph::copy_runtime_info(conv, slice);
                    padded_input_plane_filter_height_one.push_back(slice);
                }

            } else {

            }
            auto h1_conv_output = std::make_shared<opset1::Convolution>(h1_conv_input, pointwise_filters[f_x],
                Strides{ filter_stride_y, filter_stride_x }, CoordinateDiff{ 0, 0 }, CoordinateDiff{ 0, 0 }, Strides{ 1, 1 }, PadType::VALID);


        }
    }
    return is_graph_modfied;
}
