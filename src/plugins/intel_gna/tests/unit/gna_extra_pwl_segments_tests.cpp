// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "any_copy.hpp"
#include "gna_infer_request.hpp"
#include "gna_plugin.hpp"
#include "ngraph_functions/builders.hpp"

using GNAPluginNS::GNAPlugin;
namespace {

struct ConvolutionParameters {
    intel_dnn_operation_t convolution_type;
    ov::Shape input_shape;
    ov::Shape kernel;
    ov::Shape stride;
    ov::Shape dilation;
    size_t numOutChannels;
    std::vector<size_t> pad_begin_pool;
    std::vector<size_t> pad_end_pool;
};

struct PWLExtraSegmentsParamsWithConv {
    ConvolutionParameters conv_params;
    ov::element::Type precision;
    ngraph::helpers::ActivationTypes activation_type;
    size_t expected_segments_num;
    bool use_pooling;
};

const std::unordered_map<ngraph::helpers::ActivationTypes, std::string> kNGraphActivationMapForTests{
    {ngraph::helpers::ActivationTypes::Relu, "Relu"},
    {ngraph::helpers::ActivationTypes::Sigmoid, "Sigmoid"}};

const std::unordered_map<intel_dnn_operation_t, std::string> kDnnOperationMapForTests{
    {kDnnConvolutional1dOp, "kDnnConvolutional1dOp"},
    {kDnnConvolutional2dOp, "kDnnConvolutional2dOp"},
};

template <typename T>
std::string GetEnumName(const T& enum_value,
                        const std::unordered_map<T, std::string>& enum_map,
                        const std::string default_value = "") {
    auto iter = enum_map.find(enum_value);
    if (iter == enum_map.end()) {
        return default_value;
    }
    return iter->second;
}

std::string GetDnnOperationName(const intel_dnn_operation_t& enum_value, const std::string default_value = "") {
    return GetEnumName(enum_value, kDnnOperationMapForTests, default_value);
}

std::string GetNGraphActivationTypeName(const ngraph::helpers::ActivationTypes& enum_value,
                                        const std::string default_value = "") {
    return GetEnumName(enum_value, kNGraphActivationMapForTests, default_value);
}

std::ostream& operator<<(std::ostream& os, const ngraph::helpers::ActivationTypes& value) {
    os << GetNGraphActivationTypeName(value);
    return os;
}

std::ostream& operator<<(std::ostream& os, const intel_dnn_operation_t& value) {
    os << GetDnnOperationName(value);
    return os;
}

class GNAPluginForPWLExtraSegmentsTest : public GNAPlugin {
public:
    GNAPluginForPWLExtraSegmentsTest(const std::map<std::string, std::string>& config) : GNAPlugin(config) {
        gnamem.reset(new GNAPluginNS::gna_memory_float(GNAPluginNS::memory::GNAFloatAllocator{}));
        graphCompiler.setGNAMemoryPtr(gnamem);
        gnadevice.reset();
    }
    void Test(const size_t expected_segments) {
        for (const auto& component : graphCompiler.dnnComponents.components) {
            if (component.dnnComponent.operation == kDnnPiecewiselinearOp) {
                EXPECT_EQ(expected_segments, component.dnnComponent.op.pwl.num_segments);
            }
        }
    }
};

class GNAPWLExtraSegmentsTestFixture : public ::testing::TestWithParam<PWLExtraSegmentsParamsWithConv> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<PWLExtraSegmentsParamsWithConv>& obj);
    void SetUp() override;

protected:
    InferenceEngine::CNNNetwork cnn_network_;
};

std::string GNAPWLExtraSegmentsTestFixture::getTestCaseName(
    const testing::TestParamInfo<PWLExtraSegmentsParamsWithConv>& obj) {
    auto params = obj.param;
    std::ostringstream result;
    result << "ConvolutionType=" << params.conv_params.convolution_type << "_";
    result << "ActivationType=" << params.activation_type << "_";
    result << "PoolingOn=" << (params.use_pooling ? "yes" : "no");
    return result.str();
}

void GNAPWLExtraSegmentsTestFixture::SetUp() {
    auto params = GetParam();
    const auto& conv_params = params.conv_params;
    const auto& precision = params.precision;
    const auto& input_shape = conv_params.input_shape;
    const auto& kernel = conv_params.kernel;
    const auto& stride = conv_params.stride;
    const auto& pad_begin_pool = conv_params.pad_begin_pool;
    const auto& pad_end_pool = conv_params.pad_end_pool;
    ov::CoordinateDiff pad_begin(pad_begin_pool.begin(), pad_begin_pool.end());
    ov::CoordinateDiff pad_end(pad_end_pool.begin(), pad_end_pool.end());
    const auto& dilation = conv_params.dilation;
    const auto& activation_type = params.activation_type;
    const auto& use_pooling = params.use_pooling;
    std::vector<size_t> filter_shape;
    filter_shape.push_back(conv_params.numOutChannels);
    filter_shape.push_back(input_shape[1]);
    filter_shape.insert(filter_shape.end(), kernel.begin(), kernel.end());

    auto input = std::make_shared<ngraph::opset9::Parameter>(precision, input_shape);
    auto filter = ngraph::builder::makeConstant<float>(precision, filter_shape, {1.f}, true);

    auto conv = std::make_shared<ngraph::opset9::Convolution>(input, filter, stride, pad_begin, pad_end, dilation);

    auto activation = ngraph::builder::makeActivation(conv, precision, activation_type);
    std::shared_ptr<ngraph::opset9::Result> result = nullptr;

    if (use_pooling) {
        auto maxpool = ngraph::builder::makePooling(activation,
                                                    stride,
                                                    pad_begin_pool,
                                                    pad_end_pool,
                                                    kernel,
                                                    ngraph::op::RoundingType::FLOOR,
                                                    ngraph::op::PadType::VALID,
                                                    false,
                                                    ngraph::helpers::PoolingTypes::MAX);
        result = std::make_shared<ngraph::opset9::Result>(maxpool);
    } else {
        result = std::make_shared<ngraph::opset9::Result>(activation);
    }

    auto function = std::make_shared<ov::Model>(ov::ResultVector({result}),
                                                ov::ParameterVector({input}),
                                                "convolution_with_activation_exrta_segments");
    cnn_network_ = InferenceEngine::CNNNetwork(function);
}

auto kPrecision32 = ov::element::f32;

const ov::Shape kInput1D = {1, 1, 1, 8};
const ov::Shape kKernel1D = {1, 2};
const ov::Shape kStride1D = {1, 1};
const ov::Shape kDilation1D = kStride1D;
const size_t kOutChanneldsNum1D = 4;
const std::vector<size_t> kPadBegin1D = {0, 0};
const std::vector<size_t> kPadEnd1D = {0, 0};
const ConvolutionParameters kConvolutionParams1D =
    {kDnnConvolutional1dOp, kInput1D, kKernel1D, kStride1D, kDilation1D, kOutChanneldsNum1D, kPadBegin1D, kPadEnd1D};

const PWLExtraSegmentsParamsWithConv kConvolution1DReluWithoutPoolParams = {kConvolutionParams1D,
                                                                            kPrecision32,
                                                                            ngraph::helpers::ActivationTypes::Relu,
                                                                            2,
                                                                            false};
const PWLExtraSegmentsParamsWithConv kConvolution1DReluWithPoolParams = {kConvolutionParams1D,
                                                                         kPrecision32,
                                                                         ngraph::helpers::ActivationTypes::Relu,
                                                                         2,
                                                                         true};
const PWLExtraSegmentsParamsWithConv kConvolution1DSigmoidWithoutPoolParams =
    {kConvolutionParams1D, kPrecision32, ngraph::helpers::ActivationTypes::Sigmoid, 12, false};
const PWLExtraSegmentsParamsWithConv kConvolution1DSigmoidWithPoolParams = {kConvolutionParams1D,
                                                                            kPrecision32,
                                                                            ngraph::helpers::ActivationTypes::Sigmoid,
                                                                            12,
                                                                            true};

const ov::Shape kInput2D = {1, 8, 20, 16};
const ov::Shape kKernel2D = {1, 1};
const ov::Shape kStride2D = {1, 1};
const ov::Shape kDilation2D = kStride2D;
const size_t kOutChanneldsNum2D = 8;
const std::vector<size_t> kPadBegin2D = {0, 0};
const std::vector<size_t> kPadEnd2D = {0, 0};
const ConvolutionParameters kConvolutionParams2D =
    {kDnnConvolutional2dOp, kInput2D, kKernel2D, kStride2D, kDilation2D, kOutChanneldsNum2D, kPadBegin2D, kPadEnd2D};

const PWLExtraSegmentsParamsWithConv kConvolution2DReluWithoutPoolParams = {kConvolutionParams2D,
                                                                            kPrecision32,
                                                                            ngraph::helpers::ActivationTypes::Relu,
                                                                            4,
                                                                            false};
const PWLExtraSegmentsParamsWithConv kConvolution2DReluWithPoolParams = {kConvolutionParams2D,
                                                                         kPrecision32,
                                                                         ngraph::helpers::ActivationTypes::Relu,
                                                                         4,
                                                                         true};
const PWLExtraSegmentsParamsWithConv kConvolution2DSigmoidWithoutPoolParams =
    {kConvolutionParams2D, kPrecision32, ngraph::helpers::ActivationTypes::Sigmoid, 12, false};
const PWLExtraSegmentsParamsWithConv kConvolution2DSigmoidWithPoolParams = {kConvolutionParams2D,
                                                                            kPrecision32,
                                                                            ngraph::helpers::ActivationTypes::Sigmoid,
                                                                            12,
                                                                            true};

INSTANTIATE_TEST_CASE_P(GNAPWLExtraSegmentsConv1DTests,
                        GNAPWLExtraSegmentsTestFixture,
                        ::testing::Values(kConvolution1DReluWithoutPoolParams,
                                          kConvolution1DReluWithPoolParams,
                                          kConvolution1DSigmoidWithoutPoolParams,
                                          kConvolution1DSigmoidWithPoolParams),
                        GNAPWLExtraSegmentsTestFixture::getTestCaseName);

INSTANTIATE_TEST_CASE_P(GNAPWLExtraSegmentsConv2DTests,
                        GNAPWLExtraSegmentsTestFixture,
                        ::testing::Values(kConvolution2DReluWithoutPoolParams,
                                          kConvolution2DReluWithPoolParams,
                                          kConvolution2DSigmoidWithoutPoolParams,
                                          kConvolution2DSigmoidWithPoolParams),
                        GNAPWLExtraSegmentsTestFixture::getTestCaseName);

TEST_P(GNAPWLExtraSegmentsTestFixture, check_number_of_segments) {
    auto params = GetParam();
    const ov::AnyMap& gna_config = {ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_EXACT)};
    GNAPluginForPWLExtraSegmentsTest plugin(ov::any_copy(gna_config));

    EXPECT_NO_THROW(plugin.LoadNetwork(cnn_network_));
    EXPECT_NO_THROW(plugin.Test(params.expected_segments_num));
}

}  // namespace