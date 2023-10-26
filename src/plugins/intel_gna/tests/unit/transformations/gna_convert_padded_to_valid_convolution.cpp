// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset7.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/init_node_info.hpp>
#include <tuple>

#include "common_test_utils/ov_test_utils.hpp"
#include "transformations/convert_padded_to_valid_convolution.hpp"

namespace testing {

namespace {

enum class modelType {
    TranspConvTransp = 0,     /* Transpose(NHWC->NCHW) => Conv => Transpose(NCHW->NHWC) */
    TranspConvBcastAddTransp, /* Transpose(NHWC->NCHW) => Conv => Broadcasted Add (Bias) => Transpose(NCHW->NHWC) */
    TranspConvActTransp,      /* Transpose(NHWC->NCHW) => Conv => Activation Function => Transpose(NCHW->NHWC) */
    TranspConvBcastAddMaxPoolTransp, /* Transpose(NHWC->NCHW) => Conv => Broadcasted Add (Bias) => MaxPooling =>
                                        Transpose(NCHW->NHWC) (2D Max Pool case) */
    TranspConvBcastAddActTransp, /* Transpose(NHWC->NCHW) => Conv => Broadcasted Add (Bias) => Activation Function =>
                                    Transpose(NCHW->NHWC) */
    TranspConvBcastAddMaxPoolActTransp, /* Transpose(NHWC->NCHW) => Conv => Broadcasted Add (Bias) => MaxPool =>
                                           Activation Function => Transpose(NCHW->NHWC) */
    TranspConvTranspBcastAdd,           /* Transpose(NHWC->NCHW) => conv => Transpose(NCHW->NHWC) => Bias */
    TranspConvTranspBcastAddAct /* Transpose(NHWC->NCHW) => Conv => Transpose(NCHW->NHWC) => Bias => Activation Function
                                 */
};

typedef std::tuple<modelType,               // Test model
                   ngraph::PartialShape,    // Input shape
                   ngraph::Shape,           // Convolution filter shape
                   ngraph::Strides,         // Convolution stride
                   ngraph::CoordinateDiff,  // Convolution pads begin
                   ngraph::CoordinateDiff,  // Convolution pads end
                   ngraph::Strides,         // Convolution dilation
                   ngraph::Shape,           // Bias shape
                   ngraph::Strides,         // Max Pool stride
                   ngraph::Shape,           // Max Pool shape
                   ngraph::op::PadType      // Padding type
                   >
    paddedToValidConvParams;

typedef std::tuple<bool,                    // With / without Fake Quantize layers
                   paddedToValidConvParams  // Test parameters
                   >
    fqPaddedToValidConvParams;

struct ConvData {
    size_t input_height;
    size_t input_width;
    size_t input_channel_count;
    size_t pads_begin_width;
    size_t pads_begin_height;
    size_t pads_end_width;
    size_t pads_end_height;
};

void GetConvParams(std::shared_ptr<ngraph::opset7::Convolution> conv, ConvData& conv_data) {
    conv_data.input_channel_count = conv->input_value(0).get_shape()[1];
    conv_data.input_height = conv->input_value(0).get_shape()[2];
    conv_data.input_width = conv->input_value(0).get_shape()[3];
    conv_data.pads_begin_height = conv->get_pads_begin()[0];
    conv_data.pads_begin_width = conv->get_pads_begin()[1];
    conv_data.pads_end_height = conv->get_pads_end()[0];
    conv_data.pads_end_width = conv->get_pads_end()[1];
}

std::shared_ptr<ngraph::opset7::FakeQuantize> createFQ(std::shared_ptr<ngraph::Node>& in_node) {
    auto input_low = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {1});
    auto input_high = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {5});
    auto output_low = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {0});
    auto output_high = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {10});
    return std::make_shared<ngraph::opset7::FakeQuantize>(in_node, input_low, input_high, output_low, output_high, 11);
}

ngraph::Output<ngraph::Node> createBiasFQ(const ngraph::Output<ngraph::Node>& in_node,
                                          std::shared_ptr<ngraph::opset7::Constant>& bias_const,
                                          const bool& fq) {
    std::shared_ptr<ngraph::Node> bcast_add = std::make_shared<ngraph::opset7::Add>(in_node, bias_const);

    if (fq) {
        bcast_add = createFQ(bcast_add);
    }

    return bcast_add;
}

std::shared_ptr<ngraph::opset7::Result> createFunction(const bool& fq,
                                                       const modelType& model,
                                                       const ngraph::Output<ngraph::Node>& input_node,
                                                       const ngraph::Shape& filters_shape,
                                                       const ngraph::Strides& conv_stride,
                                                       const ngraph::CoordinateDiff& pads_begin,
                                                       const ngraph::CoordinateDiff& pads_end,
                                                       const ngraph::Strides& conv_dilation,
                                                       const ngraph::Shape& bias_shape,
                                                       const ngraph::Strides& maxpool_stride,
                                                       const ngraph::Shape& maxpool_shape,
                                                       const ngraph::op::PadType& pad_type,
                                                       ConvData* conv_data) {
    auto transpose_in_order = std::make_shared<ngraph::opset7::Constant>(ngraph::element::i64,
                                                                         ngraph::Shape{4},
                                                                         std::vector<int64_t>{0, 3, 1, 2});
    auto transpose_in = std::make_shared<ngraph::opset7::Transpose>(input_node, transpose_in_order);
    std::shared_ptr<ngraph::Node> filters = std::make_shared<ngraph::opset7::Constant>(
        ngraph::element::i64,
        ngraph::Shape{4, input_node.get_shape()[3], filters_shape[0], filters_shape[1]});

    if (fq) {
        filters = createFQ(filters);
    }

    auto conv = std::make_shared<ngraph::opset7::Convolution>(transpose_in,
                                                              filters,
                                                              conv_stride,
                                                              pads_begin,
                                                              pads_end,
                                                              conv_dilation,
                                                              pad_type);
    if (conv_data)
        GetConvParams(conv, *conv_data);
    auto transpose_out_order = std::make_shared<ngraph::opset7::Constant>(ngraph::element::i64,
                                                                          ngraph::Shape{4},
                                                                          std::vector<int64_t>{0, 2, 3, 1});
    auto bias_const = std::make_shared<ngraph::opset7::Constant>(ngraph::element::i64, bias_shape);

    ngraph::Output<ngraph::Node> last_op = std::make_shared<ngraph::opset7::Transpose>(conv, transpose_out_order);

    switch (model) {
    case modelType::TranspConvBcastAddTransp: {
        auto bcast_add = createBiasFQ(conv, bias_const, fq);
        last_op = std::make_shared<ngraph::opset7::Transpose>(bcast_add, transpose_out_order);
    } break;

    case modelType::TranspConvActTransp: {
        auto bcast_add = createBiasFQ(conv, bias_const, fq);
        std::shared_ptr<ngraph::Node> activation = std::make_shared<ngraph::opset7::Relu>(bcast_add);

        if (fq) {
            activation = createFQ(activation);
        }

        last_op = std::make_shared<ngraph::opset7::Transpose>(activation, transpose_out_order);
    } break;

    case modelType::TranspConvBcastAddMaxPoolTransp: {
        auto bcast_add = createBiasFQ(conv, bias_const, fq);
        auto maxpool = std::make_shared<ngraph::opset7::MaxPool>(bcast_add,
                                                                 maxpool_stride,
                                                                 ngraph::Shape{0, 0},
                                                                 ngraph::Shape{0, 0},
                                                                 maxpool_shape,
                                                                 ngraph::op::RoundingType::FLOOR,
                                                                 ngraph::op::PadType::VALID);
        auto transpose = std::make_shared<ngraph::opset7::Transpose>(maxpool, transpose_out_order);
        last_op = std::make_shared<ngraph::opset7::Relu>(transpose);
    } break;

    case modelType::TranspConvBcastAddActTransp: {
        auto bcast_add = createBiasFQ(conv, bias_const, fq);
        auto activation = std::make_shared<ngraph::opset7::Relu>(bcast_add);
        last_op = std::make_shared<ngraph::opset7::Transpose>(activation, transpose_out_order);
    } break;

    case modelType::TranspConvBcastAddMaxPoolActTransp: {
        auto bcast_add = createBiasFQ(conv, bias_const, fq);
        auto maxpool = std::make_shared<ngraph::opset7::MaxPool>(bcast_add,
                                                                 maxpool_stride,
                                                                 ngraph::Shape{0, 0},
                                                                 ngraph::Shape{0, 0},
                                                                 maxpool_shape,
                                                                 ngraph::op::RoundingType::FLOOR,
                                                                 ngraph::op::PadType::VALID);
        auto activation = std::make_shared<ngraph::opset7::Relu>(maxpool);
        last_op = std::make_shared<ngraph::opset7::Transpose>(activation, transpose_out_order);
    } break;

    case modelType::TranspConvTranspBcastAdd: {
        last_op = createBiasFQ(last_op, bias_const, fq);
    } break;

    case modelType::TranspConvTranspBcastAddAct: {
        auto bcast_add = createBiasFQ(last_op, bias_const, fq);
        last_op = std::make_shared<ngraph::opset7::Relu>(bcast_add);
    } break;

    case modelType::TranspConvTransp:
    default:
        break;
    }

    return std::make_shared<ngraph::opset7::Result>(last_op);
}

std::shared_ptr<ngraph::Function> get_initial_function(const bool& fq,
                                                       const modelType& model,
                                                       const ngraph::PartialShape& input_shape,
                                                       const ngraph::Shape& filters_shape,
                                                       const ngraph::Strides& conv_stride,
                                                       const ngraph::CoordinateDiff& pads_begin,
                                                       const ngraph::CoordinateDiff& pads_end,
                                                       const ngraph::Strides& conv_dilation,
                                                       const ngraph::Shape& bias_shape,
                                                       const ngraph::Strides& maxpool_stride,
                                                       const ngraph::Shape& maxpool_shape,
                                                       const ngraph::op::PadType& pad_type,
                                                       ConvData& conv_data) {
    auto input_params = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::i64, input_shape);
    auto result = createFunction(fq,
                                 model,
                                 input_params,
                                 filters_shape,
                                 conv_stride,
                                 pads_begin,
                                 pads_end,
                                 conv_dilation,
                                 bias_shape,
                                 maxpool_stride,
                                 maxpool_shape,
                                 pad_type,
                                 &conv_data);
    return std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{input_params});
}

// ---------------------------------------------------------------------------------------------------------------------

class ConvertPaddedToValidConvTestInvalidFixture : public ov::test::TestsCommon,
                                                   public ::testing::WithParamInterface<fqPaddedToValidConvParams> {
public:
    void SetUp() override;

public:
    std::shared_ptr<ngraph::Function> function, reference_function;
};

void ConvertPaddedToValidConvTestInvalidFixture::SetUp() {
    bool fq;
    paddedToValidConvParams params;
    modelType model;
    ngraph::PartialShape input_shape;
    ngraph::Shape filters_shape, bias_shape, maxpool_shape;
    ngraph::Strides conv_stride, conv_dilation, maxpool_stride;
    ngraph::CoordinateDiff pads_begin, pads_end;
    ngraph::op::PadType pad_type;
    ConvData conv_data;
    std::tie(fq, params) = this->GetParam();
    std::tie(model,
             input_shape,
             filters_shape,
             conv_stride,
             pads_begin,
             pads_end,
             conv_dilation,
             bias_shape,
             maxpool_stride,
             maxpool_shape,
             pad_type) = params;

    function = get_initial_function(fq,
                                    model,
                                    input_shape,
                                    filters_shape,
                                    conv_stride,
                                    pads_begin,
                                    pads_end,
                                    conv_dilation,
                                    bias_shape,
                                    maxpool_stride,
                                    maxpool_shape,
                                    pad_type,
                                    conv_data);
    reference_function = get_initial_function(fq,
                                              model,
                                              input_shape,
                                              filters_shape,
                                              conv_stride,
                                              pads_begin,
                                              pads_end,
                                              conv_dilation,
                                              bias_shape,
                                              maxpool_stride,
                                              maxpool_shape,
                                              pad_type,
                                              conv_data);
}

// ---------------------------------------------------------------------------------------------------------------------

class ConvertPaddedToValidConvTestFixture : public ov::test::TestsCommon,
                                            public ::testing::WithParamInterface<fqPaddedToValidConvParams> {
public:
    void SetUp() override;
    std::shared_ptr<ngraph::Function> get_reference(const bool& fq,
                                                    const modelType& model,
                                                    const ngraph::PartialShape& input_shape,
                                                    const ngraph::Shape& filters_shape,
                                                    const ngraph::Strides& conv_stride,
                                                    const ngraph::CoordinateDiff& pads_begin,
                                                    const ngraph::CoordinateDiff& pads_end,
                                                    const ngraph::Strides& conv_dilation,
                                                    const ngraph::Shape& bias_shape,
                                                    const ngraph::Strides& maxpool_stride,
                                                    const ngraph::Shape& maxpool_shape,
                                                    const ngraph::op::PadType& pad_type,
                                                    const ConvData& conv_data);

public:
    std::shared_ptr<ngraph::Function> function, reference_function;
};

void ConvertPaddedToValidConvTestFixture::SetUp() {
    bool fq;
    paddedToValidConvParams params;
    modelType model;
    ngraph::PartialShape input_shape;
    ngraph::Shape filters_shape, bias_shape, maxpool_shape;
    ngraph::Strides conv_stride, conv_dilation, maxpool_stride;
    ngraph::CoordinateDiff pads_begin, pads_end;
    ngraph::op::PadType pad_type;
    ConvData conv_data;
    std::tie(fq, params) = this->GetParam();
    std::tie(model,
             input_shape,
             filters_shape,
             conv_stride,
             pads_begin,
             pads_end,
             conv_dilation,
             bias_shape,
             maxpool_stride,
             maxpool_shape,
             pad_type) = params;

    function = get_initial_function(fq,
                                    model,
                                    input_shape,
                                    filters_shape,
                                    conv_stride,
                                    pads_begin,
                                    pads_end,
                                    conv_dilation,
                                    bias_shape,
                                    maxpool_stride,
                                    maxpool_shape,
                                    pad_type,
                                    conv_data);
    reference_function = get_reference(fq,
                                       model,
                                       input_shape,
                                       filters_shape,
                                       conv_stride,
                                       pads_begin,
                                       pads_end,
                                       conv_dilation,
                                       bias_shape,
                                       maxpool_stride,
                                       maxpool_shape,
                                       pad_type,
                                       conv_data);
}

std::shared_ptr<ngraph::opset7::StridedSlice> FlatCrop(ngraph::Output<ngraph::Node> input, size_t offset, size_t size) {
    return std::make_shared<ngraph::opset7::StridedSlice>(
        input,  // data
        ngraph::opset7::Constant::create(ngraph::element::i64,
                                         ngraph::Shape{2},
                                         {(size_t)0, offset}),  // begin sice index
        ngraph::opset7::Constant::create(ngraph::element::i64,
                                         ngraph::Shape{2},
                                         {(size_t)0, offset + size}),  // end slice index
        ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {(size_t)1, (size_t)1}),  // strides
        std::vector<int64_t>{1, 0},                                                                        // begin mask
        std::vector<int64_t>{1, 0});                                                                       // end mask
}

void InsertPadding(ngraph::OutputVector& input_rows_to_concat,
                   size_t size,
                   const std::shared_ptr<ngraph::opset7::Constant> padding_const,
                   size_t biggest_padding) {
    if (size == biggest_padding) {
        input_rows_to_concat.push_back(padding_const);
    } else {
        auto slice = FlatCrop(padding_const, 0, size);
        input_rows_to_concat.push_back(slice);
    }
}

std::shared_ptr<ngraph::Node> CreatePaddedNet(const ngraph::Output<ngraph::Node>& input_node,
                                              const ConvData& conv_data) {
    size_t flat_left_padding = conv_data.input_channel_count * conv_data.pads_begin_width;
    size_t flat_right_padding = conv_data.input_channel_count * conv_data.pads_end_width;
    size_t padded_row_size =
        flat_left_padding + conv_data.input_channel_count * conv_data.input_width + flat_right_padding;
    size_t flat_top_padding = padded_row_size * conv_data.pads_begin_height;
    size_t flat_bottom_padding = padded_row_size * conv_data.pads_end_height;
    size_t biggest_padding =
        std::max(std::max(flat_left_padding, flat_right_padding), std::max(flat_top_padding, flat_bottom_padding));

    if (conv_data.input_height > 1 && (flat_top_padding > 1 || flat_bottom_padding > 1)) {
        biggest_padding = biggest_padding > padded_row_size ? biggest_padding : padded_row_size;
    }

    if (!biggest_padding)
        return nullptr;

    auto flat_input = std::make_shared<ngraph::opset7::Reshape>(
        input_node,
        ngraph::opset7::Constant::create(ngraph::element::i64,
                                         ngraph::Shape{2},
                                         ngraph::Shape{1ull, shape_size(input_node.get_shape())}),
        false);

    // Constant with zero padding
    auto const_holding_padding =
        std::make_shared<ngraph::opset7::Constant>(ngraph::element::i64, ngraph::Shape{1, biggest_padding}, 0);

    std::shared_ptr<ngraph::Node> original_row = flat_input;
    ngraph::OutputVector input_rows_to_concat;

    // Add top padding
    for (size_t p = 0; p < conv_data.pads_begin_height; p++) {
        InsertPadding(input_rows_to_concat, padded_row_size, const_holding_padding, biggest_padding);
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
                original_row = FlatCrop(flat_input,
                                        h * conv_data.input_width * conv_data.input_channel_count,
                                        conv_data.input_width * conv_data.input_channel_count);

            ngraph::OutputVector single_row_concat_inputs;
            if (flat_left_padding) {
                InsertPadding(single_row_concat_inputs, flat_left_padding, const_holding_padding, biggest_padding);
            }
            single_row_concat_inputs.push_back(original_row);
            if (flat_right_padding) {
                InsertPadding(single_row_concat_inputs, flat_right_padding, const_holding_padding, biggest_padding);
            }
            auto padded_row_concat = std::make_shared<ngraph::opset7::Concat>(single_row_concat_inputs, 1);

            input_rows_to_concat.push_back(padded_row_concat);
        }
    } else {
        input_rows_to_concat.push_back(original_row);
    }

    // Bottom padding
    for (size_t p = 0; p < conv_data.pads_end_height; p++) {
        InsertPadding(input_rows_to_concat, padded_row_size, const_holding_padding, biggest_padding);
    }

    auto padded_input_plane = std::make_shared<ngraph::opset7::Concat>(input_rows_to_concat, 1);
    return padded_input_plane;
}

std::shared_ptr<ngraph::Function> ConvertPaddedToValidConvTestFixture::get_reference(
    const bool& fq,
    const modelType& model,
    const ngraph::PartialShape& input_shape,
    const ngraph::Shape& filters_shape,
    const ngraph::Strides& conv_stride,
    const ngraph::CoordinateDiff& pads_begin,
    const ngraph::CoordinateDiff& pads_end,
    const ngraph::Strides& conv_dilation,
    const ngraph::Shape& bias_shape,
    const ngraph::Strides& maxpool_stride,
    const ngraph::Shape& maxpool_shape,
    const ngraph::op::PadType& pad_type,
    const ConvData& conv_data) {
    auto input_params = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::i64, input_shape);

    // Add padding where neccessary

    // padding
    // padding
    // ... row ...
    // ... row ...
    // ...........
    // ... row ...
    // padding
    // padding
    auto padded_input_plane = CreatePaddedNet(input_params, conv_data);
    std::shared_ptr<ngraph::opset7::Result> result;

    if (padded_input_plane) {
        auto shape_const = std::make_shared<ngraph::opset7::Constant>(
            ngraph::element::i64,
            ngraph::Shape{4},
            ngraph::Shape{static_cast<size_t>(1),
                          conv_data.pads_begin_height + conv_data.input_height + conv_data.pads_end_height,
                          conv_data.pads_begin_width + conv_data.input_width + conv_data.pads_end_width,
                          conv_data.input_channel_count});
        auto padded_input_plane_reshaped =
            std::make_shared<ngraph::opset7::Reshape>(padded_input_plane, shape_const, false);
        result = createFunction(fq,
                                model,
                                padded_input_plane_reshaped,
                                filters_shape,
                                conv_stride,
                                ngraph::CoordinateDiff{0, 0},
                                ngraph::CoordinateDiff{0, 0},
                                conv_dilation,
                                bias_shape,
                                maxpool_stride,
                                maxpool_shape,
                                ngraph::op::PadType::EXPLICIT,
                                nullptr);
    } else {
        // Valid padding
        result = createFunction(fq,
                                model,
                                input_params,
                                filters_shape,
                                conv_stride,
                                pads_begin,
                                pads_end,
                                conv_dilation,
                                bias_shape,
                                maxpool_stride,
                                maxpool_shape,
                                pad_type,
                                nullptr);
    }

    return std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{input_params});
}

// ---------------------------------------------------------------------------------------------------------------------

void execute_test(std::shared_ptr<ngraph::Function> function, std::shared_ptr<ngraph::Function> reference_function) {
    ngraph::pass::Manager manager;
    manager.register_pass<ov::pass::InitNodeInfo>();
    manager.register_pass<ov::intel_gna::pass::ConvertPaddedToValidConv>();
    manager.run_passes(function);
    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(function, reference_function);
    ASSERT_TRUE(result.valid);
}

TEST_P(ConvertPaddedToValidConvTestFixture, CompareFunctions) {
    execute_test(function, reference_function);
}

INSTANTIATE_TEST_SUITE_P(ConvertPaddedToValidConvTestSuite,
                         ConvertPaddedToValidConvTestFixture,
                         ::testing::Combine(
                             // With / without Fake Quantize layers
                             ::testing::Values(true, false),
                             ::testing::Values(std::make_tuple(modelType::TranspConvTransp,
                                                               ngraph::PartialShape{1, 1, 16, 8},
                                                               ngraph::Shape{1, 2},
                                                               ngraph::Strides{1, 1},
                                                               ngraph::CoordinateDiff{0, 2},
                                                               ngraph::CoordinateDiff{0, 3},
                                                               ngraph::Strides{1, 1},
                                                               ngraph::Shape{1, 4, 1, 1},
                                                               ngraph::Strides{1, 1},
                                                               ngraph::Shape{1, 2},
                                                               ngraph::op::PadType::EXPLICIT),
                                               std::make_tuple(modelType::TranspConvBcastAddTransp,
                                                               ngraph::PartialShape{1, 1, 16, 8},
                                                               ngraph::Shape{1, 2},
                                                               ngraph::Strides{1, 1},
                                                               ngraph::CoordinateDiff{0, 2},
                                                               ngraph::CoordinateDiff{0, 3},
                                                               ngraph::Strides{1, 1},
                                                               ngraph::Shape{1, 4, 1, 1},
                                                               ngraph::Strides{1, 1},
                                                               ngraph::Shape{1, 2},
                                                               ngraph::op::PadType::EXPLICIT),
                                               std::make_tuple(modelType::TranspConvActTransp,
                                                               ngraph::PartialShape{1, 1, 16, 8},
                                                               ngraph::Shape{1, 2},
                                                               ngraph::Strides{1, 1},
                                                               ngraph::CoordinateDiff{0, 2},
                                                               ngraph::CoordinateDiff{0, 3},
                                                               ngraph::Strides{1, 1},
                                                               ngraph::Shape{1, 4, 1, 1},
                                                               ngraph::Strides{1, 1},
                                                               ngraph::Shape{1, 2},
                                                               ngraph::op::PadType::EXPLICIT),
                                               std::make_tuple(modelType::TranspConvBcastAddMaxPoolTransp,
                                                               ngraph::PartialShape{1, 1, 16, 8},
                                                               ngraph::Shape{1, 2},
                                                               ngraph::Strides{1, 1},
                                                               ngraph::CoordinateDiff{0, 2},
                                                               ngraph::CoordinateDiff{0, 3},
                                                               ngraph::Strides{1, 1},
                                                               ngraph::Shape{1, 4, 1, 1},
                                                               ngraph::Strides{1, 1},
                                                               ngraph::Shape{1, 2},
                                                               ngraph::op::PadType::EXPLICIT),
                                               std::make_tuple(modelType::TranspConvBcastAddActTransp,
                                                               ngraph::PartialShape{1, 1, 16, 8},
                                                               ngraph::Shape{1, 2},
                                                               ngraph::Strides{1, 1},
                                                               ngraph::CoordinateDiff{0, 2},
                                                               ngraph::CoordinateDiff{0, 3},
                                                               ngraph::Strides{1, 1},
                                                               ngraph::Shape{1, 4, 1, 1},
                                                               ngraph::Strides{1, 1},
                                                               ngraph::Shape{1, 2},
                                                               ngraph::op::PadType::SAME_LOWER),
                                               std::make_tuple(modelType::TranspConvBcastAddMaxPoolActTransp,
                                                               ngraph::PartialShape{1, 1, 16, 8},
                                                               ngraph::Shape{1, 2},
                                                               ngraph::Strides{1, 1},
                                                               ngraph::CoordinateDiff{0, 2},
                                                               ngraph::CoordinateDiff{0, 3},
                                                               ngraph::Strides{1, 1},
                                                               ngraph::Shape{1, 4, 1, 1},
                                                               ngraph::Strides{1, 1},
                                                               ngraph::Shape{1, 2},
                                                               ngraph::op::PadType::SAME_UPPER),
                                               std::make_tuple(modelType::TranspConvTranspBcastAdd,
                                                               ngraph::PartialShape{1, 1, 16, 8},
                                                               ngraph::Shape{1, 2},
                                                               ngraph::Strides{1, 1},
                                                               ngraph::CoordinateDiff{0, 2},
                                                               ngraph::CoordinateDiff{0, 3},
                                                               ngraph::Strides{1, 1},
                                                               ngraph::Shape{1, 1, 1, 4},
                                                               ngraph::Strides{1, 1},
                                                               ngraph::Shape{1, 2},
                                                               ngraph::op::PadType::EXPLICIT),
                                               std::make_tuple(modelType::TranspConvTranspBcastAddAct,
                                                               ngraph::PartialShape{1, 1, 16, 8},
                                                               ngraph::Shape{1, 2},
                                                               ngraph::Strides{1, 1},
                                                               ngraph::CoordinateDiff{0, 2},
                                                               ngraph::CoordinateDiff{0, 3},
                                                               ngraph::Strides{1, 1},
                                                               ngraph::Shape{1, 1, 1, 4},
                                                               ngraph::Strides{1, 1},
                                                               ngraph::Shape{1, 2},
                                                               ngraph::op::PadType::EXPLICIT))));

TEST_P(ConvertPaddedToValidConvTestInvalidFixture, CompareFunctions) {
    execute_test(function, reference_function);
}

INSTANTIATE_TEST_SUITE_P(ConvertPaddedToValidConvInvalidTestSuite,
                         ConvertPaddedToValidConvTestInvalidFixture,
                         ::testing::Combine(
                             // With / without Fake Quantize layers
                             ::testing::Values(true, false),
                             ::testing::Values(std::make_tuple(modelType::TranspConvTransp,
                                                               ngraph::PartialShape{2, 1, 16, 8},
                                                               ngraph::Shape{1, 2},
                                                               ngraph::Strides{1, 1},
                                                               ngraph::CoordinateDiff{0, 2},
                                                               ngraph::CoordinateDiff{0, 3},
                                                               ngraph::Strides{1, 1},
                                                               ngraph::Shape{1, 4, 1, 1},
                                                               ngraph::Strides{1, 1},
                                                               ngraph::Shape{1, 2},
                                                               ngraph::op::PadType::SAME_UPPER),
                                               std::make_tuple(modelType::TranspConvBcastAddTransp,
                                                               ngraph::PartialShape{2, 1, 16, 8},
                                                               ngraph::Shape{1, 2},
                                                               ngraph::Strides{1, 1},
                                                               ngraph::CoordinateDiff{0, 2},
                                                               ngraph::CoordinateDiff{0, 3},
                                                               ngraph::Strides{1, 1},
                                                               ngraph::Shape{1, 4, 1, 1},
                                                               ngraph::Strides{1, 1},
                                                               ngraph::Shape{1, 2},
                                                               ngraph::op::PadType::EXPLICIT),
                                               std::make_tuple(modelType::TranspConvActTransp,
                                                               ngraph::PartialShape{2, 1, 16, 8},
                                                               ngraph::Shape{1, 2},
                                                               ngraph::Strides{1, 1},
                                                               ngraph::CoordinateDiff{0, 2},
                                                               ngraph::CoordinateDiff{0, 3},
                                                               ngraph::Strides{1, 1},
                                                               ngraph::Shape{1, 4, 1, 1},
                                                               ngraph::Strides{1, 1},
                                                               ngraph::Shape{1, 2},
                                                               ngraph::op::PadType::EXPLICIT),
                                               std::make_tuple(modelType::TranspConvBcastAddMaxPoolTransp,
                                                               ngraph::PartialShape{2, 16, 16, 8},
                                                               ngraph::Shape{1, 2},
                                                               ngraph::Strides{1, 1},
                                                               ngraph::CoordinateDiff{0, 2},
                                                               ngraph::CoordinateDiff{0, 3},
                                                               ngraph::Strides{1, 1},
                                                               ngraph::Shape{1, 4, 1, 1},
                                                               ngraph::Strides{1, 1},
                                                               ngraph::Shape{5, 1},
                                                               ngraph::op::PadType::EXPLICIT),
                                               std::make_tuple(modelType::TranspConvBcastAddActTransp,
                                                               ngraph::PartialShape{2, 1, 16, 8},
                                                               ngraph::Shape{1, 2},
                                                               ngraph::Strides{1, 1},
                                                               ngraph::CoordinateDiff{0, 2},
                                                               ngraph::CoordinateDiff{0, 3},
                                                               ngraph::Strides{1, 1},
                                                               ngraph::Shape{1, 4, 1, 1},
                                                               ngraph::Strides{1, 1},
                                                               ngraph::Shape{1, 2},
                                                               ngraph::op::PadType::SAME_LOWER),
                                               std::make_tuple(modelType::TranspConvBcastAddMaxPoolActTransp,
                                                               ngraph::PartialShape{2, 1, 16, 8},
                                                               ngraph::Shape{1, 2},
                                                               ngraph::Strides{1, 1},
                                                               ngraph::CoordinateDiff{0, 5},
                                                               ngraph::CoordinateDiff{0, 3},
                                                               ngraph::Strides{1, 1},
                                                               ngraph::Shape{1, 4, 1, 1},
                                                               ngraph::Strides{1, 1},
                                                               ngraph::Shape{1, 4},
                                                               ngraph::op::PadType::SAME_UPPER),
                                               std::make_tuple(modelType::TranspConvTranspBcastAdd,
                                                               ngraph::PartialShape{2, 1, 16, 8},
                                                               ngraph::Shape{1, 2},
                                                               ngraph::Strides{1, 1},
                                                               ngraph::CoordinateDiff{0, 2},
                                                               ngraph::CoordinateDiff{0, 3},
                                                               ngraph::Strides{1, 1},
                                                               ngraph::Shape{1, 1, 1, 4},
                                                               ngraph::Strides{1, 1},
                                                               ngraph::Shape{1, 2},
                                                               ngraph::op::PadType::EXPLICIT),
                                               std::make_tuple(modelType::TranspConvTranspBcastAddAct,
                                                               ngraph::PartialShape{2, 1, 16, 8},
                                                               ngraph::Shape{1, 2},
                                                               ngraph::Strides{1, 1},
                                                               ngraph::CoordinateDiff{0, 2},
                                                               ngraph::CoordinateDiff{0, 3},
                                                               ngraph::Strides{1, 1},
                                                               ngraph::Shape{1, 1, 1, 4},
                                                               ngraph::Strides{1, 1},
                                                               ngraph::Shape{1, 2},
                                                               ngraph::op::PadType::EXPLICIT))));

}  // namespace

}  // namespace testing
