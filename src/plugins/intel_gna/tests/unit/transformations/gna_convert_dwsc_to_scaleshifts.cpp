// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset7.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include <tuple>

#include "common_test_utils/ov_test_utils.hpp"
#include "transformations/convert_dwsc_to_scaleshifts.hpp"

namespace testing {

namespace {

enum class modelType {
    TranspDWSCTransp = 0, /* Transpose(NHWC->NCHW) => DWSC (Group Convolution) => Transpose(NCHW->NHWC) */
    TranspDWSCBiasTransp, /* Transpose(NHWC->NCHW) => DWSC => Broadcasted Add (Bias) => Transpose(NCHW->NHWC) */
};

typedef std::tuple<modelType,               // Test model
                   ngraph::Shape,           // Input shape
                   ngraph::Shape,           // Convolution filter shape
                   ngraph::Strides,         // Convolution stride
                   ngraph::CoordinateDiff,  // Convolution pads begin
                   ngraph::CoordinateDiff,  // Convolution pads end
                   ngraph::Strides,         // Convolution dilation
                   ngraph::Shape,           // Bias shape
                   ngraph::op::PadType      // Padding type
                   >
    DWSCToScaleShiftsParams;

typedef std::tuple<bool,                    // With / without Fake Quantize layers
                   DWSCToScaleShiftsParams  // Test parameters
                   >
    fqDWSCToScaleShiftsParams;

std::shared_ptr<ngraph::opset7::FakeQuantize> createFQ(std::shared_ptr<ngraph::Node>& in_node) {
    auto input_low = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {1});
    auto input_high = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {5});
    auto output_low = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {0});
    auto output_high = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {10});
    return std::make_shared<ngraph::opset7::FakeQuantize>(in_node, input_low, input_high, output_low, output_high, 11);
}

std::shared_ptr<ngraph::Node> createBiasFQ(const std::shared_ptr<ngraph::Node>& in_node,
                                           std::shared_ptr<ngraph::opset7::Constant>& bias_const,
                                           const bool& fq) {
    std::shared_ptr<ngraph::Node> node;
    node = std::make_shared<ngraph::opset7::Add>(in_node, bias_const);

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
                                                       const ngraph::CoordinateDiff& pads_begin,
                                                       const ngraph::CoordinateDiff& pads_end,
                                                       const ngraph::Strides& conv_dilation,
                                                       const ngraph::Shape& bias_shape,
                                                       const ngraph::op::PadType& pad_type,
                                                       std::shared_ptr<ngraph::opset7::GroupConvolution>& dwsc,
                                                       std::shared_ptr<ngraph::opset7::Constant>& bias_const,
                                                       std::shared_ptr<ngraph::opset7::FakeQuantize>& fq_bias) {
    std::shared_ptr<ngraph::Node> fq_filters;

    auto transpose_in_order = std::make_shared<ngraph::opset7::Constant>(ngraph::element::i64,
                                                                         ngraph::Shape{4},
                                                                         std::vector<int64_t>{0, 3, 1, 2});
    auto transpose_in = std::make_shared<ngraph::opset7::Transpose>(input_node, transpose_in_order);

    if (fq) {
        fq_filters = std::make_shared<ngraph::opset7::Constant>(
            ngraph::element::i64,
            ngraph::Shape{input_node.get_shape()[3], 1, filters_shape[0], filters_shape[1]});
        fq_filters = createFQ(fq_filters);
        fq_filters = std::make_shared<ngraph::opset7::Reshape>(
            fq_filters,
            ngraph::opset7::Constant::create(
                ngraph::element::i64,
                ngraph::Shape{5},
                ngraph::Shape{input_node.get_shape()[3], 1, 1, filters_shape[0], filters_shape[1]}),
            false);
    } else {
        fq_filters = std::make_shared<ngraph::opset7::Constant>(
            ngraph::element::i64,
            ngraph::Shape{input_node.get_shape()[3], 1, 1, filters_shape[0], filters_shape[1]});
    }

    dwsc = std::make_shared<ngraph::opset7::GroupConvolution>(transpose_in,
                                                              fq_filters,
                                                              conv_stride,
                                                              pads_begin,
                                                              pads_end,
                                                              conv_dilation,
                                                              pad_type);
    auto transpose_out_order = std::make_shared<ngraph::opset7::Constant>(ngraph::element::i64,
                                                                          ngraph::Shape{4},
                                                                          std::vector<int64_t>{0, 2, 3, 1});
    auto last_op = std::make_shared<ngraph::opset7::Transpose>(dwsc, transpose_out_order);

    if (model == modelType::TranspDWSCBiasTransp || fq) {
        bias_const = std::make_shared<ngraph::opset7::Constant>(ngraph::element::i64, bias_shape);
        auto bias = createBiasFQ(dwsc, bias_const, fq);
        fq_bias = std::dynamic_pointer_cast<ngraph::opset7::FakeQuantize>(bias);
        last_op = std::make_shared<ngraph::opset7::Transpose>(bias, transpose_out_order);
    }

    return std::make_shared<ngraph::opset7::Result>(last_op);
}

std::shared_ptr<ngraph::Function> get_initial_function(const bool& fq,
                                                       const modelType& model,
                                                       const ngraph::Shape& input_shape,
                                                       const ngraph::Shape& filters_shape,
                                                       const ngraph::Strides& conv_stride,
                                                       const ngraph::CoordinateDiff& pads_begin,
                                                       const ngraph::CoordinateDiff& pads_end,
                                                       const ngraph::Strides& conv_dilation,
                                                       const ngraph::Shape& bias_shape,
                                                       const ngraph::op::PadType& pad_type,
                                                       std::shared_ptr<ngraph::opset7::GroupConvolution>& dwsc,
                                                       std::shared_ptr<ngraph::opset7::Constant>& bias_const,
                                                       std::shared_ptr<ngraph::opset7::FakeQuantize>& fq_bias) {
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
                                 pad_type,
                                 dwsc,
                                 bias_const,
                                 fq_bias);
    return std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{input_params});
}

// ---------------------------------------------------------------------------------------------------------------------

class ConvertDWSCToScaleShiftsTestInvalidFixture : public ov::test::TestsCommon,
                                                   public ::testing::WithParamInterface<fqDWSCToScaleShiftsParams> {
public:
    void SetUp() override;

public:
    std::shared_ptr<ngraph::Function> function, reference_function;
    modelType model;
};

void ConvertDWSCToScaleShiftsTestInvalidFixture::SetUp() {
    bool fq;
    DWSCToScaleShiftsParams params;
    ngraph::Shape input_shape;
    ngraph::Shape filters_shape, bias_shape;
    ngraph::Strides conv_stride, conv_dilation;
    ngraph::CoordinateDiff pads_begin, pads_end;
    ngraph::op::PadType pad_type;
    std::shared_ptr<ngraph::opset7::GroupConvolution> dwsc;
    std::shared_ptr<ngraph::opset7::Constant> bias_const;
    std::shared_ptr<ngraph::opset7::FakeQuantize> fq_bias;
    std::tie(fq, params) = this->GetParam();
    std::tie(model,
             input_shape,
             filters_shape,
             conv_stride,
             pads_begin,
             pads_end,
             conv_dilation,
             bias_shape,
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
                                    pad_type,
                                    dwsc,
                                    bias_const,
                                    fq_bias);
    reference_function = get_initial_function(fq,
                                              model,
                                              input_shape,
                                              filters_shape,
                                              conv_stride,
                                              pads_begin,
                                              pads_end,
                                              conv_dilation,
                                              bias_shape,
                                              pad_type,
                                              dwsc,
                                              bias_const,
                                              fq_bias);
}

// ---------------------------------------------------------------------------------------------------------------------

class ConvertDWSCToScaleShiftsTestFixture : public ov::test::TestsCommon,
                                            public ::testing::WithParamInterface<fqDWSCToScaleShiftsParams> {
public:
    void SetUp() override;
    std::shared_ptr<ngraph::Function> get_reference(const bool& fq,
                                                    const modelType& model,
                                                    const ngraph::Shape& input_shape,
                                                    const ngraph::Shape& filters_shape,
                                                    const ngraph::Strides& conv_stride,
                                                    const ngraph::CoordinateDiff& pads_begin,
                                                    const ngraph::CoordinateDiff& pads_end,
                                                    const ngraph::Strides& conv_dilation,
                                                    const ngraph::Shape& bias_shape,
                                                    const ngraph::op::PadType& pad_type,
                                                    const std::shared_ptr<ngraph::opset7::GroupConvolution>& dwsc,
                                                    const std::shared_ptr<ngraph::opset7::Constant>& bias_const,
                                                    const std::shared_ptr<ngraph::opset7::FakeQuantize>& fq_bias);

public:
    std::shared_ptr<ngraph::Function> function, reference_function;
    modelType model;
};

void ConvertDWSCToScaleShiftsTestFixture::SetUp() {
    bool fq;
    DWSCToScaleShiftsParams params;
    ngraph::Shape input_shape;
    ngraph::Shape filters_shape, bias_shape;
    ngraph::Strides conv_stride, conv_dilation;
    ngraph::CoordinateDiff pads_begin, pads_end;
    ngraph::op::PadType pad_type;
    std::shared_ptr<ngraph::opset7::GroupConvolution> dwsc;
    std::shared_ptr<ngraph::opset7::Constant> bias_const;
    std::shared_ptr<ngraph::opset7::FakeQuantize> fq_bias;
    std::tie(fq, params) = this->GetParam();
    std::tie(model,
             input_shape,
             filters_shape,
             conv_stride,
             pads_begin,
             pads_end,
             conv_dilation,
             bias_shape,
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
                                    pad_type,
                                    dwsc,
                                    bias_const,
                                    fq_bias);
    reference_function = get_reference(fq,
                                       model,
                                       input_shape,
                                       filters_shape,
                                       conv_stride,
                                       pads_begin,
                                       pads_end,
                                       conv_dilation,
                                       bias_shape,
                                       pad_type,
                                       dwsc,
                                       bias_const,
                                       fq_bias);
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

std::shared_ptr<ngraph::Node> InsertFQLayer(const std::shared_ptr<ngraph::opset7::FakeQuantize> fq_layer,
                                            std::shared_ptr<ngraph::Node> last_node) {
    if (fq_layer != nullptr) {
        return fq_layer->clone_with_new_inputs(
            {last_node,
             ngraph::opset7::Constant::create(
                 ngraph::element::f32,
                 ngraph::Shape{1},
                 std::dynamic_pointer_cast<ngraph::opset7::Constant>(fq_layer->input_value(1).get_node_shared_ptr())
                     ->cast_vector<float>()),
             ngraph::opset7::Constant::create(
                 ngraph::element::f32,
                 ngraph::Shape{1},
                 std::dynamic_pointer_cast<ngraph::opset7::Constant>(fq_layer->input_value(2).get_node_shared_ptr())
                     ->cast_vector<float>()),
             ngraph::opset7::Constant::create(
                 ngraph::element::f32,
                 ngraph::Shape{1},
                 std::dynamic_pointer_cast<ngraph::opset7::Constant>(fq_layer->input_value(3).get_node_shared_ptr())
                     ->cast_vector<float>()),
             ngraph::opset7::Constant::create(
                 ngraph::element::f32,
                 ngraph::Shape{1},
                 std::dynamic_pointer_cast<ngraph::opset7::Constant>(fq_layer->input_value(4).get_node_shared_ptr())
                     ->cast_vector<float>())});
    }
    return last_node;
}

std::shared_ptr<ngraph::Node> DecomposeDWSC(std::shared_ptr<ngraph::opset7::GroupConvolution> dwsc,
                                            std::shared_ptr<ngraph::opset7::Constant> bias_const,
                                            std::shared_ptr<ngraph::opset7::FakeQuantize> fq_bias,
                                            std::shared_ptr<ngraph::opset7::Reshape> flat_input_plane,
                                            std::shared_ptr<ngraph::Node> flat_filters_plane) {
    std::shared_ptr<ngraph::opset7::Constant> const_zero_padding;
    std::shared_ptr<ngraph::Node> reshaped_bias;
    ngraph::OutputVector output_chunks;
    auto input_channel_count = dwsc->get_input_shape(0)[1];
    auto input_width = dwsc->get_input_shape(0)[3];
    auto output_width = dwsc->get_output_shape(0)[3];
    auto filter_width = dwsc->get_input_shape(1)[4];
    auto pads_begin = dwsc->get_pads_begin()[1];
    auto stride_width = dwsc->get_strides()[1];
    auto dilation_width = dwsc->get_dilations()[1];

    // Constant with zero padding
    if (pads_begin) {
        const_zero_padding = std::make_shared<ngraph::opset7::Constant>(dwsc->get_element_type(),
                                                                        ngraph::Shape{1, input_channel_count},
                                                                        0);
    }

    // Reshape bias const
    if (bias_const) {
        auto bias_size = shape_size(bias_const->get_shape());
        reshaped_bias = ov::op::util::make_try_fold<ngraph::opset7::Reshape>(
            bias_const,
            ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{2}, ngraph::Shape{1, bias_size}),
            false);
    }

    // Move filter over input performing multiplication and addition (scaleshift), take padding, stride, dilation and
    // bias into account
    for (int32_t input_position = -pads_begin, o = 0; o < output_width; input_position += stride_width, o++) {
        std::shared_ptr<ngraph::Node> previous_layer_output, last_layer_output;
        int32_t filter_end = input_position + filter_width * dilation_width;
        bool first = true;

        filter_end = filter_end < input_width ? filter_end : input_width;

        for (int32_t filter_pos = input_position, filter_idx = 0; filter_pos < filter_end;
             filter_pos += dilation_width, filter_idx++) {
            if (filter_pos >= 0) {
                auto conv_input_slice =
                    FlatCrop(flat_input_plane, filter_pos * input_channel_count, input_channel_count);
                auto conv_filter_slice =
                    FlatCrop(flat_filters_plane, filter_idx * input_channel_count, input_channel_count);

                if (first) {
                    first = false;
                    previous_layer_output =
                        std::make_shared<ngraph::opset7::Multiply>(conv_input_slice, conv_filter_slice);
                    if (bias_const) {
                        previous_layer_output =
                            std::make_shared<ngraph::opset7::Add>(previous_layer_output, reshaped_bias);
                        previous_layer_output = InsertFQLayer(fq_bias, previous_layer_output);
                    }
                    last_layer_output = previous_layer_output;
                } else {
                    last_layer_output = std::make_shared<ngraph::opset7::Multiply>(conv_input_slice, conv_filter_slice);
                    last_layer_output = std::make_shared<ngraph::opset7::Add>(last_layer_output, previous_layer_output);
                    previous_layer_output = last_layer_output;
                }
            }
        }

        if (!last_layer_output) {
            IE_ASSERT(const_zero_padding);
            last_layer_output = const_zero_padding;
        }

        output_chunks.push_back(last_layer_output);
    }

    // Concat and transpose is only needed when output width > 1
    if (output_chunks.size() > 1) {
        return std::make_shared<ngraph::opset7::Concat>(output_chunks, 0);
    }

    return output_chunks[0].get_node_shared_ptr();
}

std::shared_ptr<ngraph::Function> ConvertDWSCToScaleShiftsTestFixture::get_reference(
    const bool& fq,
    const modelType& model,
    const ngraph::Shape& input_shape,
    const ngraph::Shape& filters_shape,
    const ngraph::Strides& conv_stride,
    const ngraph::CoordinateDiff& pads_begin,
    const ngraph::CoordinateDiff& pads_end,
    const ngraph::Strides& conv_dilation,
    const ngraph::Shape& bias_shape,
    const ngraph::op::PadType& pad_type,
    const std::shared_ptr<ngraph::opset7::GroupConvolution>& dwsc,
    const std::shared_ptr<ngraph::opset7::Constant>& bias_const,
    const std::shared_ptr<ngraph::opset7::FakeQuantize>& fq_bias) {
    auto input_params = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::i64, input_shape);
    auto output_channel_count = dwsc->get_output_shape(0)[1];
    auto output_width = dwsc->get_output_shape(0)[3];

    // Prepare flat input data
    auto flat_input_plane = std::make_shared<ngraph::opset7::Reshape>(
        input_params,
        ngraph::opset7::Constant::create(ngraph::element::i64,
                                         ngraph::Shape{2},
                                         ngraph::Shape{1, ngraph::shape_size(input_shape)}),
        false);

    // Prepare flat filter data
    auto filters_const = std::dynamic_pointer_cast<ngraph::Node>(dwsc->get_input_node_shared_ptr(1));
    auto filters_size = ngraph::shape_size(filters_const->get_shape());

    auto transposed_filters_const = ov::op::util::make_try_fold<ngraph::opset7::Transpose>(
        filters_const,
        ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{5}, ngraph::Shape{4, 1, 2, 3, 0}));

    auto flat_filters_plane = ov::op::util::make_try_fold<ngraph::opset7::Reshape>(
        transposed_filters_const,
        ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{2}, ngraph::Shape{1, filters_size}),
        false);

    // Convert DWSC to a set of diagonal layers
    auto output_plane = DecomposeDWSC(dwsc, bias_const, fq_bias, flat_input_plane, flat_filters_plane);

    // Restore the original output shape
    auto result = std::make_shared<ngraph::opset7::Reshape>(
        output_plane,
        ngraph::opset7::Constant::create(ngraph::element::i64,
                                         ngraph::Shape{4},
                                         ngraph::Shape{1, output_channel_count, 1, output_width}),
        false);

    return std::make_shared<ngraph::Function>(ngraph::ResultVector{std::make_shared<ngraph::opset7::Result>(result)},
                                              ngraph::ParameterVector{input_params});
}

// ---------------------------------------------------------------------------------------------------------------------

void execute_test(modelType model,
                  std::shared_ptr<ngraph::Function> function,
                  std::shared_ptr<ngraph::Function> reference_function) {
    ngraph::pass::Manager manager;
    manager.register_pass<ov::pass::InitNodeInfo>();

    manager.register_pass<ov::intel_gna::pass::ConvertDWSCToScaleShifts>();
    manager.run_passes(function);
    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(function, reference_function);
    ASSERT_TRUE(result.valid);
}

TEST_P(ConvertDWSCToScaleShiftsTestFixture, CompareFunctions) {
    execute_test(model, function, reference_function);
}

INSTANTIATE_TEST_SUITE_P(ConvertDWSCToScaleShiftsTestSuite,
                         ConvertDWSCToScaleShiftsTestFixture,
                         ::testing::Combine(
                             // With / without Fake Quantize layers
                             ::testing::Values(true, false),
                             ::testing::Values(std::make_tuple(modelType::TranspDWSCTransp,
                                                               ngraph::Shape{1, 1, 5, 32},
                                                               ngraph::Shape{1, 3},
                                                               ngraph::Strides{1, 1},
                                                               ngraph::CoordinateDiff{0, 1},
                                                               ngraph::CoordinateDiff{0, 1},
                                                               ngraph::Strides{1, 1},
                                                               ngraph::Shape{1, 32, 1, 1},
                                                               ngraph::op::PadType::VALID),
                                               std::make_tuple(modelType::TranspDWSCBiasTransp,
                                                               ngraph::Shape{1, 1, 5, 32},
                                                               ngraph::Shape{1, 3},
                                                               ngraph::Strides{1, 1},
                                                               ngraph::CoordinateDiff{0, 2},
                                                               ngraph::CoordinateDiff{0, 2},
                                                               ngraph::Strides{1, 1},
                                                               ngraph::Shape{1, 32, 1, 1},
                                                               ngraph::op::PadType::VALID))));

TEST_P(ConvertDWSCToScaleShiftsTestInvalidFixture, CompareFunctions) {
    execute_test(model, function, reference_function);
}

INSTANTIATE_TEST_SUITE_P(ConvertDWSCToScaleShiftsInvalidTestSuite,
                         ConvertDWSCToScaleShiftsTestInvalidFixture,
                         ::testing::Combine(
                             // With / without Fake Quantize layers
                             ::testing::Values(true, false),
                             ::testing::Values(std::make_tuple(modelType::TranspDWSCTransp,
                                                               ngraph::Shape{2, 16, 8, 1},
                                                               ngraph::Shape{1, 2},
                                                               ngraph::Strides{1, 1},
                                                               ngraph::CoordinateDiff{0, 2},
                                                               ngraph::CoordinateDiff{0, 3},
                                                               ngraph::Strides{1, 1},
                                                               ngraph::Shape{1, 4, 1, 1},
                                                               ngraph::op::PadType::SAME_UPPER),
                                               std::make_tuple(modelType::TranspDWSCBiasTransp,
                                                               ngraph::Shape{2, 16, 8, 1},
                                                               ngraph::Shape{1, 2},
                                                               ngraph::Strides{1, 1},
                                                               ngraph::CoordinateDiff{0, 2},
                                                               ngraph::CoordinateDiff{0, 3},
                                                               ngraph::Strides{1, 1},
                                                               ngraph::Shape{1, 4, 1, 1},
                                                               ngraph::op::PadType::EXPLICIT))));

}  // namespace

}  // namespace testing
