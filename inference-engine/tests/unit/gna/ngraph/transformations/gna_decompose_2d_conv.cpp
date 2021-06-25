// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <tuple>

#include "transformations/decompose_2d_conv.hpp"
#include "common_test_utils/ngraph_test_utils.hpp"
#include <ngraph/function.hpp>
#include <ngraph/opsets/opset7.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/init_node_info.hpp>
#include "backend/gna_limitations.hpp"

namespace testing {

namespace {

enum class modelType {
    TranspConvTransp = 0,               /* Transpose(NHWC->NCHW) => Conv => Transpose(NCHW->NHWC) */
    TranspConvBcastAddTransp,           /* Transpose(NHWC->NCHW) => Conv => Broadcasted Add (Bias) => Transpose(NCHW->NHWC) */
    TranspConvBcastAddMaxPoolTransp,    /* Transpose(NHWC->NCHW) => Conv => Broadcasted Add (Bias) => MaxPooling => Transpose(NCHW->NHWC) (2D Max Pool case) */
    TranspConvBcastAddActTransp,        /* Transpose(NHWC->NCHW) => Conv => Broadcasted Add (Bias) => Activation Function => Transpose(NCHW->NHWC) */
    TranspConvBcastAddMaxPoolActTransp, /* Transpose(NHWC->NCHW) => Conv => Broadcasted Add (Bias) => MaxPool => Activation Function => Transpose(NCHW->NHWC) */
    TranspConvTranspBcastAdd,           /* Transpose(NHWC->NCHW) => conv => Transpose(NCHW->NHWC) => Bias */
    TranspConvTranspBcastAddAct         /* Transpose(NHWC->NCHW) => Conv => Transpose(NCHW->NHWC) => Bias => Activation Function */
};

typedef std::tuple<
    modelType,              // Test model
    ngraph::PartialShape,   // Input shape
    ngraph::Shape,          // Convolution filter shape
    ngraph::Strides,        // Convolution stride
    ngraph::Strides,        // Convolution dilation
    ngraph::Shape,          // Bias shape
    ngraph::Strides,        // Max Pool stride
    ngraph::Shape           // Max Pool shape
> decompose2DConvParams;

typedef std::tuple<
    bool,                   // With / without Fake Quantize layers
    decompose2DConvParams   // Test parameters
> fqDecompose2DConvParams;

struct GraphData {
    std::shared_ptr<ngraph::Node> input_node;
    std::shared_ptr<ngraph::opset7::Convolution> conv;
    std::shared_ptr<ngraph::opset7::Add> bias;
    std::shared_ptr<ngraph::op::util::UnaryElementwiseArithmetic> af;
    std::shared_ptr<ngraph::opset7::MaxPool> max_pool;
    std::shared_ptr<ngraph::Node> bias_const;
    std::shared_ptr<ngraph::Node> last_op_in_sequence_for_replacement;
    size_t conv_count;
    size_t pool_size_width;
    size_t pool_stride_width;
};

struct ConvParams {
    size_t input_height;
    size_t input_width;
    size_t input_channel_count;
    size_t output_channel_count;
    size_t filter_height;
    size_t filter_width;
    size_t filter_count;
    size_t filter_channel_count;
    size_t filter_dilation_height;
    size_t filter_dilation_width;
    size_t filter_stride_height;
    size_t filter_stride_width;
    size_t output_height;
    size_t output_width;
};

void GetConvParams(std::shared_ptr<ngraph::opset7::Convolution> conv, ConvParams& conv_params) {
    conv_params.output_height = conv->get_output_shape(0)[2];
    conv_params.output_width = conv->get_output_shape(0)[3];
    conv_params.input_channel_count = conv->input_value(0).get_shape()[1];
    conv_params.input_height = conv->input_value(0).get_shape()[2];
    conv_params.input_width = conv->input_value(0).get_shape()[3];
    conv_params.filter_count = conv->input_value(1).get_shape()[0];
    conv_params.filter_channel_count = conv->input_value(1).get_shape()[1];
    conv_params.filter_height = conv->input_value(1).get_shape()[2];
    conv_params.filter_width = conv->input_value(1).get_shape()[3];
    conv_params.filter_dilation_height = conv->get_dilations()[0];
    conv_params.filter_dilation_width = conv->get_dilations()[1];
    conv_params.filter_stride_height = conv->get_strides()[0];
    conv_params.filter_stride_width = conv->get_strides()[1];
    conv_params.output_channel_count = conv_params.filter_count;
}

std::shared_ptr<ngraph::opset7::FakeQuantize> createFQ(std::shared_ptr<ngraph::Node>& in_node) {
    auto input_low = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {1});
    auto input_high = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {5});
    auto output_low = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {0});
    auto output_high = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {10});
    return std::make_shared<ngraph::opset7::FakeQuantize>(in_node, input_low, input_high, output_low, output_high, 11);
}

std::shared_ptr<ngraph::Node> createBiasFQ(const ngraph::Output<ngraph::Node>& in_node,
    std::shared_ptr<ngraph::opset7::Constant>& bias_const, std::shared_ptr<ngraph::opset7::Add>& bias, const bool& fq) {
    std::shared_ptr<ngraph::Node> node;
    bias = std::make_shared<ngraph::opset7::Add>(in_node, bias_const);
    node = bias;

    if (fq) {
        node = createFQ(node);
    }

    return node;
}

std::shared_ptr<ngraph::opset7::Result> createFunction(const bool& fq,
    const modelType& model,
    const ngraph::Output<ngraph::Node>& input_node,
    const ngraph::Shape& filters_shape,
    const ngraph::Strides& conv_stride,
    const ngraph::Strides& conv_dilation,
    const ngraph::Shape& bias_shape,
    const ngraph::Strides& maxpool_stride,
    const ngraph::Shape& maxpool_shape,
    GraphData* graph_data,
    ConvParams* conv_params) {
    auto transpose_in_order = std::make_shared<ngraph::opset7::Constant>(ngraph::element::i64, ngraph::Shape{4}, std::vector<int64_t>{0, 3, 1, 2});
    auto transpose_in = std::make_shared<ngraph::opset7::Transpose>(input_node, transpose_in_order);
    std::shared_ptr<ngraph::Node> filters = std::make_shared<ngraph::opset7::Constant>(ngraph::element::i64,
        ngraph::Shape{4, input_node.get_shape()[3], filters_shape[0], filters_shape[1]});

    if (fq) {
        filters = createFQ(filters);
    }

    auto conv = std::make_shared<ngraph::opset7::Convolution>(transpose_in, filters, conv_stride,
        ngraph::CoordinateDiff{0, 0}, ngraph::CoordinateDiff{0, 0}, conv_dilation, ngraph::op::PadType::VALID);
    if (conv_params)
        GetConvParams(conv, *conv_params);

    auto transpose_out_order = std::make_shared<ngraph::opset7::Constant>(ngraph::element::i64, ngraph::Shape{4}, std::vector<int64_t>{0, 2, 3, 1});
    auto bias_const = std::make_shared<ngraph::opset7::Constant>(ngraph::element::i64, bias_shape);
    std::shared_ptr<ngraph::opset7::Add> bias = nullptr;
    std::shared_ptr<ngraph::opset7::MaxPool> max_pool = nullptr;
    std::shared_ptr<ngraph::op::util::UnaryElementwiseArithmetic> activation = nullptr;
    std::shared_ptr<ngraph::Node> last_op = std::make_shared<ngraph::opset7::Transpose>(conv, transpose_out_order);

    switch (model) {
    case modelType::TranspConvBcastAddTransp:
    {
        auto bias_fq = createBiasFQ(conv, bias_const, bias, fq);
        last_op = std::make_shared<ngraph::opset7::Transpose>(bias_fq, transpose_out_order);
    }
    break;

    case modelType::TranspConvBcastAddMaxPoolTransp:
    {
        auto bias_fq = createBiasFQ(conv, bias_const, bias, fq);
        max_pool = std::make_shared<ngraph::opset7::MaxPool>(bias_fq, maxpool_stride, ngraph::Shape{0, 0}, ngraph::Shape{0, 0}, maxpool_shape,
            ngraph::op::RoundingType::FLOOR, ngraph::op::PadType::VALID);
        auto transpose = std::make_shared<ngraph::opset7::Transpose>(max_pool, transpose_out_order);
        last_op = std::make_shared<ngraph::opset7::Relu>(transpose);
    }
    break;

    case modelType::TranspConvBcastAddActTransp:
    {
        auto bias_fq = createBiasFQ(conv, bias_const, bias, fq);
        activation = std::make_shared<ngraph::opset7::Relu>(bias_fq);
        last_op = std::make_shared<ngraph::opset7::Transpose>(activation, transpose_out_order);
    }
    break;

    case modelType::TranspConvBcastAddMaxPoolActTransp:
    {
        auto bias_fq = createBiasFQ(conv, bias_const, bias, fq);
        max_pool = std::make_shared<ngraph::opset7::MaxPool>(bias_fq, maxpool_stride, ngraph::Shape{0, 0}, ngraph::Shape{0, 0}, maxpool_shape,
            ngraph::op::RoundingType::FLOOR, ngraph::op::PadType::VALID);
        activation = std::make_shared<ngraph::opset7::Relu>(max_pool);
        last_op = std::make_shared<ngraph::opset7::Transpose>(activation, transpose_out_order);
    }
    break;

    case modelType::TranspConvTranspBcastAdd:
    {
        last_op = createBiasFQ(last_op, bias_const, bias, fq);
    }
    break;

    case modelType::TranspConvTranspBcastAddAct:
    {
        auto bias_fq = createBiasFQ(last_op, bias_const, bias, fq);
        last_op = std::make_shared<ngraph::opset7::Relu>(bias_fq);
    }
    break;

    case modelType::TranspConvTransp:
    default:
        break;
    }

    if (graph_data) {
        graph_data->conv = conv;
        graph_data->bias = bias;
        graph_data->af = activation;
        graph_data->max_pool = max_pool;
        graph_data->last_op_in_sequence_for_replacement = last_op;
        graph_data->bias_const = nullptr;
        graph_data->conv_count = 0;

        if (max_pool) {
            graph_data->pool_size_width = max_pool->get_kernel()[1];
            graph_data->pool_stride_width = max_pool->get_strides()[1];
        }
    }

    return std::make_shared<ngraph::opset7::Result>(last_op);
}

std::shared_ptr<ngraph::Function> get_initial_function(const bool& fq,
    const modelType& model,
    const ngraph::PartialShape& input_shape,
    const ngraph::Shape& filters_shape,
    const ngraph::Strides& conv_stride,
    const ngraph::Strides& conv_dilation,
    const ngraph::Shape& bias_shape,
    const ngraph::Strides& maxpool_stride,
    const ngraph::Shape& maxpool_shape,
    GraphData& graph_data,
    ConvParams& conv_params) {
    auto input_params = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::i64, input_shape);
    auto result = createFunction(fq, model, input_params, filters_shape, conv_stride, conv_dilation, bias_shape,
        maxpool_stride, maxpool_shape, &graph_data , &conv_params);
    return std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{input_params});
}

// ---------------------------------------------------------------------------------------------------------------------

class Decompose2DConvTestInvalidFixture : public CommonTestUtils::TestsCommon,
    public ::testing::WithParamInterface<fqDecompose2DConvParams> {
public:
    void SetUp() override;
public:
    std::shared_ptr<ngraph::Function> function, reference_function;
    modelType model;
};

void Decompose2DConvTestInvalidFixture::SetUp() {
    bool fq;
    decompose2DConvParams params;
    ngraph::PartialShape input_shape;
    ngraph::Shape filters_shape, bias_shape, maxpool_shape;
    ngraph::Strides conv_stride, conv_dilation, maxpool_stride;
    GraphData graph_data{};
    ConvParams conv_params{};
    std::tie(fq, params) = this->GetParam();
    std::tie(model, input_shape, filters_shape, conv_stride, conv_dilation,
        bias_shape, maxpool_stride, maxpool_shape) = params;

    function = get_initial_function(fq, model, input_shape, filters_shape, conv_stride, conv_dilation,
        bias_shape, maxpool_stride, maxpool_shape, graph_data, conv_params);
    reference_function = get_initial_function(fq, model, input_shape, filters_shape, conv_stride, conv_dilation,
        bias_shape, maxpool_stride, maxpool_shape, graph_data, conv_params);
}

// ---------------------------------------------------------------------------------------------------------------------

class Decompose2DConvTestFixture: public CommonTestUtils::TestsCommon,
    public ::testing::WithParamInterface<fqDecompose2DConvParams> {
public:
    void SetUp() override;
    std::shared_ptr<ngraph::Function> get_reference(const bool& fq,
        const modelType& model,
        const ngraph::PartialShape& input_shape,
        GraphData& graph_data,
        ConvParams& conv_params);
public:
    std::shared_ptr<ngraph::Function> function, reference_function;
    modelType model;
};

void Decompose2DConvTestFixture::SetUp() {
    bool fq;
    decompose2DConvParams params;
    ngraph::PartialShape input_shape;
    ngraph::Shape filters_shape, bias_shape, maxpool_shape;
    ngraph::Strides conv_stride, conv_dilation, maxpool_stride;
    GraphData graph_data{};
    ConvParams conv_params{};
    std::tie(fq, params) = this->GetParam();
    std::tie(model, input_shape, filters_shape, conv_stride, conv_dilation,
        bias_shape, maxpool_stride, maxpool_shape) = params;

    function = get_initial_function(fq, model, input_shape, filters_shape, conv_stride, conv_dilation,
        bias_shape, maxpool_stride, maxpool_shape, graph_data, conv_params);
    reference_function = get_reference(fq, model, input_shape, graph_data, conv_params);
}

std::shared_ptr<ngraph::Node> CreateBiasConst(std::shared_ptr<ngraph::opset7::Add> conv_bias, const ConvParams& conv_params) {
    auto add_const = std::dynamic_pointer_cast<ngraph::op::Constant>(conv_bias->input_value(1).get_node_shared_ptr());

    IE_ASSERT(add_const);

    auto bias_size = shape_size(add_const->get_shape());
    const auto* srd_data_pointer = add_const->get_data_ptr<float>();
    std::vector<float> bias_values(srd_data_pointer, srd_data_pointer + bias_size);
    return ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{1, bias_size, 1, 1}, bias_values);
}

std::shared_ptr<ngraph::opset7::StridedSlice> FlatCrop(ngraph::Output<ngraph::Node> input, size_t offset, size_t size) {
    auto shape = input.get_shape();
    return std::make_shared<ngraph::opset7::StridedSlice>(
        input,                                                                                                  // data
        ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {(size_t)0, offset}),          // begin slice index
        ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {(size_t)0, offset + size}),   // end slice index
        ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {(size_t)1, (size_t)1}),       // strides
        std::vector<int64_t>{1, 0},                                                                             // begin mask
        std::vector<int64_t>{1, 0});                                                                            // end mask
}

static std::vector<std::shared_ptr<ngraph::opset7::Constant>> Split2DConvFilters(std::shared_ptr<ngraph::opset7::Constant>& filters,
    const bool& vertical_permute, const bool& horizontal_permute, const size_t& split_channels) {

    std::vector <std::shared_ptr<ngraph::opset7::Constant>> result;
    auto filter_shape = filters->get_output_shape(0);
    if (!horizontal_permute && !vertical_permute && split_channels == 1)
        return {filters};

    IE_ASSERT(filter_shape.size() == 4);

    std::vector<std::vector<int64_t>> flat_filters;
    flat_filters.resize(split_channels);
    for (size_t i = 0; i < split_channels; i++)
        flat_filters[i].resize(shape_size(filter_shape) / split_channels);

    auto N = filter_shape[0];
    auto C = filter_shape[1];
    auto H = filter_shape[2];
    auto W = filter_shape[3];

    size_t CS = (C / split_channels);
    const auto* data = filters->get_data_ptr<ngraph::element::i64>();

    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < CS; c++) {
            for (size_t s = 0; s < split_channels; s++) {
                for (size_t h = 0; h < H; h++) {
                    for (size_t w = 0; w < W; w++) {
                        if (vertical_permute || !horizontal_permute) {
                            flat_filters[s][n * CS * H * W + c * H * W + h * W + w] =
                                data[n * C * H * W + (c * split_channels + s) * H * W + h * W + w];
                        } else if (horizontal_permute) {
                            flat_filters[s][n * CS * H * W + c * H * W + H * w + h] =
                                data[n * C * H * W + (c * split_channels + s) * H * W + h * W + w];
                        }
                    }
                }
            }
        }
    }

    if (vertical_permute && horizontal_permute) {
        for (auto new_filter : flat_filters)
            result.push_back(std::make_shared<ngraph::opset7::Constant>(ngraph::element::i64,
                ngraph::Shape{filter_shape[0], filter_shape[1] * filter_shape[2] * filter_shape[3] / split_channels, 1, 1}, new_filter));
    } else if (vertical_permute && !horizontal_permute) {
        for (auto new_filter : flat_filters)
            result.push_back(std::make_shared<ngraph::opset7::Constant>(ngraph::element::i64,
                ngraph::Shape{filter_shape[0], filter_shape[1] * filter_shape[2] / split_channels, 1, filter_shape[3]}, new_filter));
    } else if (!vertical_permute && horizontal_permute) {
        for (auto new_filter : flat_filters)
            result.push_back(std::make_shared<ngraph::opset7::Constant>(ngraph::element::i64,
                ngraph::Shape{filter_shape[0], filter_shape[1] * filter_shape[3] / split_channels, filter_shape[2], 1}, new_filter));
    } else {
        for (auto new_filter : flat_filters)
            result.push_back(std::make_shared<ngraph::opset7::Constant>(ngraph::element::i64,
                ngraph::Shape{filter_shape[0], filter_shape[1] / split_channels, filter_shape[2], filter_shape[3]}, new_filter));
    }

    return result;
}

ngraph::OutputVector SplitInput(const GraphData& graph_data, ConvParams& conv_params) {
    // We need to have proper input shape first
    ngraph::OutputVector split_planes;
    auto padded_input_plane = std::make_shared<ngraph::opset7::Reshape>(graph_data.input_node,
        ngraph::op::Constant::create(ngraph::element::i64, ngraph::Shape{2},
            ngraph::Shape{1ull, shape_size(graph_data.input_node->get_shape())}), false);

    if (graph_data.conv_count > 1) {
        // If we split input plane and filters due to GNA limitations - we must sum their results at the end
        conv_params.input_channel_count /= graph_data.conv_count;

        auto reshape_before_transpose = std::make_shared<ngraph::opset7::Reshape>(padded_input_plane,
            ngraph::op::Constant::create(ngraph::element::i64, ngraph::Shape{2},
                {shape_size(padded_input_plane->get_shape()) / graph_data.conv_count, graph_data.conv_count}), false);

        auto transpose_before_channel_wise_split = std::make_shared<ngraph::opset7::Transpose>(reshape_before_transpose,
            ngraph::op::Constant::create(ngraph::element::Type_t::i64, ngraph::Shape{2}, {1ll, 0ll})->output(0));

        const auto axis_node = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{}, {0});
        const auto split = std::make_shared<ngraph::opset7::Split>(transpose_before_channel_wise_split, axis_node, graph_data.conv_count);
        split_planes = split->outputs();
    } else {
        split_planes.push_back(padded_input_plane);
    }

    return split_planes;
}

std::vector<std::shared_ptr<ngraph::opset7::Constant>> SplitFilters(const GraphData& graph_data, ConvParams& conv_params) {
    // If the input plane exceeds GNA limits and we have split into several convolutions, then we need to split filter data as well;
    // we also need to take filter height and potential dilation into account when modifying the filters
    auto filter_values = std::dynamic_pointer_cast<ngraph::opset7::Constant>(graph_data.conv->input_value(1).get_node_shared_ptr());
    bool vertical_permute = (conv_params.filter_height > 1);
    bool horizontal_permute = (conv_params.filter_dilation_width > 1);
    std::vector<std::shared_ptr<ngraph::opset7::Constant>> h_1_filters{};

    h_1_filters = Split2DConvFilters(filter_values, vertical_permute, horizontal_permute, graph_data.conv_count);

    return h_1_filters;
}

void TransformInput(const GraphData& graph_data, const ConvParams& conv_params, ngraph::Output<ngraph::Node>& split_input_plane) {
    /*
    *              Padded row - NHWC order
    *                  |
    *        Split in vertical dim (filter height)
    *                / | \
    *                Concat
    *                  |
    *              Transpose
    */

    // First we need to prepare flat (height = 1) slices of input data proper for flattened (height = 1) filter size
    ngraph::OutputVector dilated_input_planes;
    for (size_t filter_height = 0; filter_height < conv_params.filter_height; filter_height++) {
        size_t offset = filter_height * conv_params.filter_dilation_height * conv_params.input_width * conv_params.input_channel_count;
        auto slice = FlatCrop(split_input_plane, offset, conv_params.input_width * conv_params.input_channel_count * conv_params.output_height);
        dilated_input_planes.push_back(slice);
    }

    // Interleaving dilated input planes
    auto dilated_chunks_concat = std::make_shared<ngraph::opset7::Concat>(dilated_input_planes, 0);

    auto transposed_dilated_chunks = std::make_shared<ngraph::op::Transpose>(dilated_chunks_concat,
        ngraph::op::Constant::create(ngraph::element::Type_t::i64, ngraph::Shape{2}, {1ll, 0ll})->output(0));

    // Flattening of interleaved input planes
    auto flattened_dilated_transposed_input = std::make_shared<ngraph::opset7::Reshape>(transposed_dilated_chunks,
        ngraph::op::Constant::create(ngraph::element::i64, ngraph::Shape{2},
            {(size_t)1, conv_params.input_width * conv_params.input_channel_count * conv_params.output_height * conv_params.filter_height}), false);

    split_input_plane = flattened_dilated_transposed_input;
}

std::shared_ptr<ngraph::Node> Create1DConv(const GraphData& graph_data, const ConvParams& conv_params, const ngraph::Output<ngraph::Node>& input,
    std::shared_ptr<ngraph::Node> filters, const size_t conv_index, const size_t h_index) {
        // Transpose NHWC => NCHW
        std::shared_ptr<ngraph::Node> nchw_input = std::make_shared<ngraph::op::Transpose>(input,
            ngraph::op::Constant::create(ngraph::element::Type_t::i64, ngraph::Shape{4}, {0ll, 3ll, 1ll, 2ll})->output(0));

        // 1D Convolution
        auto conv = std::make_shared<ngraph::opset7::Convolution>(nchw_input, filters,
            ngraph::Strides{1, conv_params.filter_stride_width}, ngraph::CoordinateDiff{0, 0}, ngraph::CoordinateDiff{0, 0},
            ngraph::Strides{1, 1}, ngraph::op::PadType::VALID);
        std::string conv_name = graph_data.conv->get_friendly_name() + "_H_" + std::to_string(h_index) + "_CH_" + std::to_string(0);
        conv->set_friendly_name(conv_name);

        // Bias
        std::shared_ptr<ngraph::Node> last_conv_block_op = conv;
        if (graph_data.bias_const && conv_index == 0) {
            last_conv_block_op = std::make_shared<ngraph::opset7::Add>(conv, graph_data.bias_const);
        }

        // Max pooling
        if (graph_data.pool_size_width > 1 || graph_data.pool_stride_width > 1) {
            last_conv_block_op = std::make_shared<ngraph::opset7::MaxPool>(last_conv_block_op, ngraph::Strides{1, graph_data.pool_stride_width},
                ngraph::Shape{0, 0}, ngraph::Shape{0, 0}, ngraph::Shape{1, graph_data.pool_size_width}, graph_data.max_pool->get_rounding_type(),
                ngraph::op::PadType::VALID);
        }
        // Activation function
        if (graph_data.af && graph_data.conv_count == 1) {
            auto af_result = graph_data.af->copy_with_new_inputs({last_conv_block_op});
            last_conv_block_op = af_result;
        }

        // Transpose NCHW => NHWC
        auto nhwc_output = std::make_shared<ngraph::op::Transpose>(last_conv_block_op,
            ngraph::op::Constant::create(ngraph::element::Type_t::i64, ngraph::Shape{4}, {0ull, 2ull, 3ull, 1ull})->output(0));
        return nhwc_output;
}

std::shared_ptr<ngraph::Node> CreateDeomposedConv(const GraphData& graph_data, ConvParams& conv_params,
    ngraph::Output<ngraph::Node>& reduced_input_plane, const std::vector<std::shared_ptr<ngraph::opset7::Constant>>& h_1_filters, const size_t conv_index) {
    ngraph::OutputVector result_chunks;
    std::shared_ptr<ngraph::Node> last_op;
    bool horizontal_permute = (conv_params.filter_dilation_width > 1);
    size_t h_1_filter_channel_count = (conv_params.input_channel_count * conv_params.filter_height);

    for (size_t output_height = 0; output_height < conv_params.output_height; output_height += conv_params.filter_stride_height) {
        size_t offset = output_height * conv_params.input_width * h_1_filter_channel_count;
        auto row = (conv_params.output_height == 1) ? reduced_input_plane :
            FlatCrop(reduced_input_plane, offset, conv_params.input_width * h_1_filter_channel_count);
        /*
            *              Padded row
            *                  |
            *        ??? <Dilation !=1> ???
            *                  |
            *         Split in vertical dim
            *                / | \
            *                Concat
            *                  |
            *               Permute
            *                  |
            *              Transpose (NHWC => NCHW)
            *                  |
            *                1D Conv (Bias | MaxPooling)
            *                  |
            *              Transpose (NCHW => NHWC)
            */
        auto nhwc_conv_y_input = row;

        if (horizontal_permute) {
            // Horizontal split - transform input accordingly
            ngraph::OutputVector dilated_chunks;
            for (size_t filter_width = 0; filter_width < conv_params.filter_width; filter_width++) {
                size_t offset = filter_width * conv_params.filter_dilation_width * h_1_filter_channel_count;
                auto slice = FlatCrop(row, offset, h_1_filter_channel_count * conv_params.output_width);
                dilated_chunks.push_back(slice);
            }

            auto dilated_chunks_concat = std::make_shared<ngraph::opset7::Concat>(dilated_chunks, 0);

            auto transposed_dilated_chunks = std::make_shared<ngraph::op::Transpose>(dilated_chunks_concat,
                ngraph::op::Constant::create(ngraph::element::Type_t::i64, ngraph::Shape{2}, {1ll, 0ll})->output(0));

            auto flatten_dilated_conv_input = std::make_shared<ngraph::opset7::Reshape>(transposed_dilated_chunks,
                ngraph::op::Constant::create(ngraph::element::i64, ngraph::Shape{4},
                    ngraph::Shape{1ull, 1ull, conv_params.output_width, h_1_filter_channel_count * conv_params.filter_width}), false);

            nhwc_conv_y_input = flatten_dilated_conv_input;
        } else {
            // If no horizontal split is done, only reshape is required before decomposed convolution
            nhwc_conv_y_input = std::make_shared<ngraph::opset7::Reshape>(nhwc_conv_y_input,
                ngraph::op::Constant::create(ngraph::element::Type_t::i64, ngraph::Shape{4},
                    ngraph::Shape{1ull, 1ull, conv_params.input_width, h_1_filter_channel_count}), false);
        }

        // Pointwise convolutions
        // Valid 1D convolution wrapped with transposes NHWC => NCHW => Conv => NCHW => NHWC
        // Activation function can be fused with convolution only if it isn't split
        auto nhwc_y_output = Create1DConv(graph_data, conv_params, nhwc_conv_y_input, h_1_filters[conv_index], conv_index, output_height);
        result_chunks.push_back(nhwc_y_output);
        last_op = nhwc_y_output;
    }

    // Horizontal dimemsion greater than 1
    if (result_chunks.size() > 1) {
        // Concat in horizontal dimension
        // In NHWC index of H is 1
        auto concatenated_sub_results = std::make_shared<ngraph::opset7::Concat>(result_chunks, 1);
        last_op = concatenated_sub_results;
    }
    return last_op;
}

static bool ShouldDecompose(GraphData& graph_data, const ConvParams& conv_params) {
    // Check if split of plane due to GNA HW limitations of 768 filter elements is possible
    graph_data.conv_count = 1;
    size_t total_factorized_conv_channel_count = (conv_params.input_channel_count * conv_params.filter_height * conv_params.filter_width);
    while (total_factorized_conv_channel_count / graph_data.conv_count > GNAPluginNS::GNALimitations::convFilterMaxSize ||
        total_factorized_conv_channel_count % graph_data.conv_count != 0 || conv_params.filter_channel_count % graph_data.conv_count != 0)
        graph_data.conv_count++;

    // Concat (copy) layer limitation allows to split up to a certain limit
    // Currently we are able to split only convolutions without pooling in horizontal dimension
    if (graph_data.conv_count > GNAPluginNS::GNALimitations::copyMaxGrouping ||
        ((graph_data.pool_size_width > 1 || graph_data.pool_stride_width > 1) && graph_data.conv_count > 1))
        return false;

    // GNA supported features - there is no need to decompose such convolution
    if (graph_data.conv_count == 1 && (conv_params.input_height == 1 || conv_params.input_width == 1) &&
        conv_params.filter_dilation_width == 1 && conv_params.filter_dilation_height == 1)
        return false;

    return true;
}

std::shared_ptr<ngraph::opset7::Result> Decompose(const GraphData& graph_data, ConvParams& conv_params) {
    std::vector<std::shared_ptr<ngraph::Node>> partial_conv_results;

    // Split input and filters due to GNA filter element count limit
    auto split_planes = SplitInput(graph_data, conv_params);
    auto h_1_filters = SplitFilters(graph_data, conv_params);

    // Do transformations in each of the splits created above
    for (size_t conv_index = 0; conv_index < graph_data.conv_count; conv_index++) {
        ngraph::Output<ngraph::Node>& split_input_plane = split_planes[conv_index];

        // Input data needs to be prepared before 2D convolution decomposition
        if (conv_params.filter_height > 1) {
            TransformInput(graph_data, conv_params, split_input_plane);
        }

        auto flat_conv = CreateDeomposedConv(graph_data, conv_params, split_input_plane, h_1_filters, conv_index);
        partial_conv_results.push_back(flat_conv);
    }

    std::shared_ptr<ngraph::Node> conv_result = partial_conv_results.front();
    for (size_t i = 1; i < partial_conv_results.size(); i++) {
        auto add_result = std::make_shared<ngraph::opset7::Add>(partial_conv_results[i], conv_result);
        conv_result = add_result;
    }

    // Activation function after trailing Transpose NCHW->NHWC
    if (graph_data.af && graph_data.conv_count > 1) {
        auto af_result = graph_data.af->copy_with_new_inputs({conv_result});
        conv_result = af_result;
    }
    // We need to put the same name as before for the Convolution layer, so its output can be used as network result
    std::string conv_result_name = graph_data.last_op_in_sequence_for_replacement->get_friendly_name();
    replace_node(graph_data.last_op_in_sequence_for_replacement, conv_result);
    conv_result->set_friendly_name(conv_result_name);

    return std::make_shared<ngraph::opset7::Result>(conv_result);
}

std::shared_ptr<ngraph::Function> Decompose2DConvTestFixture::get_reference(const bool& fq,
    const modelType& model,
    const ngraph::PartialShape& input_shape,
    GraphData& graph_data,
    ConvParams& conv_params) {
    auto input_params = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::i64, input_shape);
    graph_data.input_node = input_params;

    ShouldDecompose(graph_data, conv_params);

    if (model != modelType::TranspConvTransp) {
        graph_data.bias_const = CreateBiasConst(std::dynamic_pointer_cast<ngraph::opset7::Add>(graph_data.bias), conv_params);
    }

    // Create decomposed reference function
    std::shared_ptr<ngraph::opset7::Result> result;
    result = Decompose(graph_data, conv_params);
    return std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{input_params});
}

// ---------------------------------------------------------------------------------------------------------------------

void execute_test(modelType model, std::shared_ptr<ngraph::Function> function, std::shared_ptr<ngraph::Function> reference_function) {
    ngraph::pass::Manager manager;
    manager.register_pass<ngraph::pass::InitNodeInfo>();

    switch (model) {
    default:
    case modelType::TranspConvTransp:
    case modelType::TranspConvBcastAddTransp:
    case modelType::TranspConvBcastAddMaxPoolTransp:
    case modelType::TranspConvBcastAddActTransp:
    case modelType::TranspConvBcastAddMaxPoolActTransp:
        manager.register_pass<GNAPluginNS::Decompose2DConv>();
    case modelType::TranspConvTranspBcastAdd:
        manager.register_pass<GNAPluginNS::Decompose2DConvTransposedWithBias>();
    case modelType::TranspConvTranspBcastAddAct:
        manager.register_pass<GNAPluginNS::Decompose2DConvTransposedWithBiasAF>();
    }

    manager.run_passes(function);
    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(function, reference_function);
    ASSERT_TRUE(result.valid);
}

TEST_P(Decompose2DConvTestFixture, CompareFunctions) {
    execute_test(model, function, reference_function);
}

INSTANTIATE_TEST_SUITE_P(Decompose2DConvTestSuite, Decompose2DConvTestFixture,
    ::testing::Combine(
        // With / without Fake Quantize layers
        ::testing::Values(false),
        ::testing::Values(
            std::make_tuple(modelType::TranspConvTransp, ngraph::PartialShape{1, 4, 4, 32}, ngraph::Shape{1, 2}, ngraph::Strides{1, 1},
                ngraph::Strides{1, 1}, ngraph::Shape{1, 4, 1, 1}, ngraph::Strides{1, 1}, ngraph::Shape{1, 1}),
            std::make_tuple(modelType::TranspConvBcastAddTransp, ngraph::PartialShape{1, 4, 4, 32}, ngraph::Shape{1, 2}, ngraph::Strides{1, 1},
                ngraph::Strides{1, 1}, ngraph::Shape{1, 4, 1, 1}, ngraph::Strides{1, 1}, ngraph::Shape{1, 1}),
            std::make_tuple(modelType::TranspConvBcastAddMaxPoolTransp, ngraph::PartialShape{1, 4, 4, 32}, ngraph::Shape{1, 2}, ngraph::Strides{1, 1},
                ngraph::Strides{1, 1}, ngraph::Shape{1, 4, 1, 1}, ngraph::Strides{1, 1}, ngraph::Shape{1, 1}),
            std::make_tuple(modelType::TranspConvBcastAddActTransp, ngraph::PartialShape{1, 4, 4, 32}, ngraph::Shape{1, 2}, ngraph::Strides{1, 1},
                ngraph::Strides{1, 1}, ngraph::Shape{1, 4, 1, 1}, ngraph::Strides{1, 1}, ngraph::Shape{1, 1}),
            std::make_tuple(modelType::TranspConvBcastAddMaxPoolActTransp, ngraph::PartialShape{1, 4, 4, 32}, ngraph::Shape{1, 2}, ngraph::Strides{1, 1},
                ngraph::Strides{1, 1}, ngraph::Shape{1, 4, 1, 1}, ngraph::Strides{1, 1}, ngraph::Shape{1, 1}),
            std::make_tuple(modelType::TranspConvTranspBcastAdd, ngraph::PartialShape{1, 4, 4, 32}, ngraph::Shape{1, 2}, ngraph::Strides{1, 1},
                ngraph::Strides{1, 1}, ngraph::Shape{1, 1, 1, 4}, ngraph::Strides{1, 1}, ngraph::Shape{1, 1}),
            std::make_tuple(modelType::TranspConvTranspBcastAddAct, ngraph::PartialShape{1, 4, 4, 32}, ngraph::Shape{1, 2}, ngraph::Strides{1, 1},
                ngraph::Strides{1, 1}, ngraph::Shape{1, 1, 1, 4}, ngraph::Strides{1, 1}, ngraph::Shape{1, 1}))));

TEST_P(Decompose2DConvTestInvalidFixture, CompareFunctions) {
    execute_test(model, function, reference_function);
}

INSTANTIATE_TEST_SUITE_P(Decompose2DConvInvalidTestSuite, Decompose2DConvTestInvalidFixture,
    ::testing::Combine(
        // With / without Fake Quantize layers
        ::testing::Values(false),
        ::testing::Values(
            std::make_tuple(modelType::TranspConvTransp, ngraph::PartialShape{1, 1, 4, 8}, ngraph::Shape{1, 2}, ngraph::Strides{1, 1},
                ngraph::Strides{1, 1}, ngraph::Shape{1, 4, 1, 1}, ngraph::Strides{1, 1}, ngraph::Shape{1, 2}),
            std::make_tuple(modelType::TranspConvBcastAddTransp, ngraph::PartialShape{2, 4, 4, 32}, ngraph::Shape{1, 2}, ngraph::Strides{1, 1},
                ngraph::Strides{1, 1}, ngraph::Shape{1, 4, 1, 1}, ngraph::Strides{1, 1}, ngraph::Shape{1, 2}),
            std::make_tuple(modelType::TranspConvBcastAddMaxPoolTransp, ngraph::PartialShape{1, 16, 16, 128}, ngraph::Shape{5, 5}, ngraph::Strides{1, 1},
                ngraph::Strides{1, 1}, ngraph::Shape{1, 4, 1, 1}, ngraph::Strides{1, 1}, ngraph::Shape{2, 2}),
            std::make_tuple(modelType::TranspConvBcastAddActTransp, ngraph::PartialShape{2, 4, 4, 32}, ngraph::Shape{1, 2}, ngraph::Strides{1, 1},
                ngraph::Strides{1, 1}, ngraph::Shape{1, 4, 1, 1}, ngraph::Strides{1, 1}, ngraph::Shape{1, 2}),
            std::make_tuple(modelType::TranspConvBcastAddMaxPoolActTransp, ngraph::PartialShape{1, 16, 16, 128}, ngraph::Shape{4, 4}, ngraph::Strides{1, 1},
                ngraph::Strides{1, 1}, ngraph::Shape{1, 4, 1, 1}, ngraph::Strides{2, 2}, ngraph::Shape{1, 2}),
            std::make_tuple(modelType::TranspConvTranspBcastAdd, ngraph::PartialShape{2, 4, 4, 32}, ngraph::Shape{1, 2}, ngraph::Strides{1, 1},
                ngraph::Strides{1, 1}, ngraph::Shape{1, 1, 1, 4}, ngraph::Strides{1, 1}, ngraph::Shape{1, 2}),
            std::make_tuple(modelType::TranspConvTranspBcastAddAct, ngraph::PartialShape{2, 4, 4, 32}, ngraph::Shape{1, 2}, ngraph::Strides{1, 1},
                ngraph::Strides{1, 1}, ngraph::Shape{1, 1, 1, 4}, ngraph::Strides{1, 1}, ngraph::Shape{1, 2}))));

} // namespace

} // namespace testing
