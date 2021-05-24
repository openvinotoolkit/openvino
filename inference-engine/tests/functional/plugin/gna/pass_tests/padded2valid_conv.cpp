// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_common.hpp"
#include <string>
#include <sstream>
#include <fstream>
#include <memory>
#include <queue>
#include <map>

#include "transformations/init_node_info.hpp"
#include "ngraph_functions/builders.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "../shared_tests_instances/skip_tests_check.hpp"

using namespace ngraph;
using namespace ngraph::opset1;

namespace LayerTestsDefinitions {

enum class modelType {
    TranspConvTransp = 0,               /* Transpose(NHWC->NCHW) => conv => Transpose(NCHW->NHWC) */
    TranspConvBcastAddTransp,           /* Transpose(NHWC->NCHW) => conv => broadcasted add (BIAS) => Transpose(NCHW->NHWC) */
    TranspConvBcastAddMaxPoolTransp,    /* Transpose(NHWC->NCHW) => conv => broadcasted add (BIAS) => MaxPooling => Transpose(NCHW->NHWC) (2d max pool case) */
    TranspConvBcastAddActTransp,        /* Transpose(NHWC->NCHW) => conv => broadcasted add (BIAS) => ActivationFunction => Transpose(NCHW->NHWC) */
    TranspConvBcastAddMaxPoolActTransp, /* Transpose(NHWC->NCHW) => conv => broadcasted add (BIAS) => MaxPool => ActivationFunction => Transpose(NCHW->NHWC) */
    TranspConvTranspBcastAdd,           /* Transpose(NHWC->NCHW) => conv => Transpose(NCHW->NHWC) => BIAS (output of MO --disable_nhwc_to_nchw option) */
    TranspConvTranspBcastAddAct         /* Transpose(NHWC->NCHW) => conv => Transpose(NCHW->NHWC) => BIAS => AF (output of MO --disable_nhwc_to_nchw option) */
};

typedef std::tuple<
    InferenceEngine::SizeVector,    // Kernel size
    InferenceEngine::SizeVector,    // Strides
    std::vector<ptrdiff_t>,         // Pad begin
    std::vector<ptrdiff_t>,         // Pad end
    InferenceEngine::SizeVector,    // Dilation
    size_t,                         // Num out channels
    op::PadType,                    // Padding type
    InferenceEngine::SizeVector,    // Bias
    InferenceEngine::SizeVector,    // Transposed Bias
    InferenceEngine::SizeVector     // Maxpool
> convSpecificParams;

typedef std::tuple<
    convSpecificParams,                 // Convolution parameters
    InferenceEngine::Precision,         // Network Precision
    std::string,                        // Target Device
    std::map<std::string, std::string>, // Configuration
    InferenceEngine::SizeVector,        // Input shapes
    modelType                           // Test model
> padded2ValidParams;

class Padded2ValidConvTest : public testing::WithParamInterface<padded2ValidParams>,
    virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<padded2ValidParams> obj) {
        convSpecificParams convParams;
        InferenceEngine::Precision netPrecision;
        std::string targetDevice;
        std::map<std::string, std::string> configuration;
        InferenceEngine::SizeVector inputShapes;
        modelType model;
        std::tie(convParams, netPrecision, targetDevice, configuration, inputShapes, model) = obj.param;
        op::PadType padType;
        InferenceEngine::SizeVector kernel, stride, dilation, bias, transpBias, maxpool;
        std::vector<ptrdiff_t> padBegin, padEnd;
        size_t convInput;
        std::tie(kernel, stride, padBegin, padEnd, dilation, convInput, padType, bias, transpBias, maxpool) = convParams;

        std::ostringstream result;
        result << "M=" << static_cast<uint32_t>(model) << "_";
        result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
        result << "K" << CommonTestUtils::vec2str(kernel) << "_";
        result << "S" << CommonTestUtils::vec2str(stride) << "_";
        result << "PB" << CommonTestUtils::vec2str(padBegin) << "_";
        result << "PE" << CommonTestUtils::vec2str(padEnd) << "_";
        result << "D=" << CommonTestUtils::vec2str(dilation) << "_";
        result << "O=" << convInput << "_";
        result << "AP=" << padType << "_";
        result << "B=" << CommonTestUtils::vec2str(bias) << "_";
        result << "B=" << CommonTestUtils::vec2str(transpBias) << "_";
        result << "MP=" << CommonTestUtils::vec2str(maxpool) << "_";
        result << "netPRC=" << netPrecision.name() << "_";
        result << "targetDevice=" << targetDevice << "_";
        for (auto const& configItem : configuration) {
            result << "_configItem=" << configItem.first << "_" << configItem.second;
        }
        return result.str();
    }

protected:
    void SetUp() override {
        convSpecificParams convParams;
        InferenceEngine::Precision netPrecision;
        std::vector<size_t> inputShape;
        modelType model;
        std::tie(convParams, netPrecision, targetDevice, configuration, inputShape, model) = this->GetParam();
        op::PadType padType;
        InferenceEngine::SizeVector kernel, stride, dilation, bias, transpBias, maxpool;
        std::vector<ptrdiff_t> padBegin, padEnd;
        size_t numOutChannels;
        std::tie(kernel, stride, padBegin, padEnd, dilation, numOutChannels, padType, bias, transpBias, maxpool) = convParams;
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

        Shape bias_shape{ bias };
        Shape transp_bias_shape{ transpBias };
        Shape maxpool_shape{ maxpool };
        std::vector<float> bias_weights{};

        auto input = builder::makeParams(ngPrc, { inputShape });
        auto transpose_in_order = op::Constant::create(element::i64, Shape{ 4 }, { 0, 3, 1, 2 });
        auto transpose_in = std::make_shared<Transpose>(input[0], transpose_in_order);
        auto filter_size = std::accumulate(std::begin(kernel), std::end(kernel), 1, std::multiplies<size_t>());
        auto filter_weights = CommonTestUtils::generate_float_numbers(numOutChannels * inputShape[3] * filter_size, -0.5f, 0.5f);
        auto conv = builder::makeConvolution(transpose_in, ngPrc, kernel, stride, padBegin,
            padEnd, dilation, padType, numOutChannels, false, filter_weights);
        auto transpose_out_order = op::Constant::create(element::i64, Shape{ 4 }, { 0, 2, 3, 1 });
        auto bias_const = builder::makeConstant(ngPrc, bias_shape, bias_weights, true);
        std::shared_ptr<Node> last_op = std::make_shared<Transpose>(conv, transpose_out_order);;

        switch (model) {
        case modelType::TranspConvBcastAddTransp:
        {
            auto bias = std::make_shared<Add>(conv, bias_const);
            last_op = std::make_shared<Transpose>(bias, transpose_out_order);
        }
        break;

        case modelType::TranspConvBcastAddMaxPoolTransp:
        {
            auto bcast_add = std::make_shared<Add>(conv, bias_const);
            auto maxpool = std::make_shared<MaxPool>(bcast_add, Strides{ 1, 1 }, Shape{ 0, 0 }, Shape{ 0, 0 }, maxpool_shape);
            last_op = std::make_shared<Transpose>(maxpool, transpose_out_order);
        }
        break;

        case modelType::TranspConvBcastAddActTransp:
        {
            auto bcast_add = std::make_shared<Add>(conv, bias_const);
            auto activation = std::make_shared<Relu>(bcast_add);
            last_op = std::make_shared<Transpose>(activation, transpose_out_order);
        }
        break;

        case modelType::TranspConvBcastAddMaxPoolActTransp:
        {
            auto bcast_add = std::make_shared<Add>(conv, bias_const);
            auto max_pool = std::make_shared<MaxPool>(bcast_add, Strides{ 1, 1 }, Shape{ 0, 0 }, Shape{ 0, 0 }, maxpool_shape);
            auto activation = std::make_shared<Relu>(max_pool);
            last_op = std::make_shared<Transpose>(activation, transpose_out_order);
        }
        break;

        case modelType::TranspConvTranspBcastAdd:
        {
            bias_const = std::make_shared<Constant>(ngPrc, transp_bias_shape);
            last_op = std::make_shared<Add>(last_op, bias_const);
        }
        break;

        case modelType::TranspConvTranspBcastAddAct:
        {
            bias_const = builder::makeConstant(ngPrc, transp_bias_shape, bias_weights, true);
            auto bcast_add = std::make_shared<Add>(last_op, bias_const);
            last_op = std::make_shared<Relu>(bcast_add);
        }
        break;

        case modelType::TranspConvTransp:
        default:
            break;
        }

        function = std::make_shared<Function>(NodeVector{ last_op }, ParameterVector{ input });
    }
};

class Gna30Padded2ValidConvTest : public Padded2ValidConvTest, GnaLayerTestCheck {
protected:
    void Run() override {
        GnaLayerTestCheck::SkipTestCheck();

        if (!GnaLayerTestCheck::skipTest) {
            Padded2ValidConvTest::Run();
        }
    }

    void SetUp() override {
        Padded2ValidConvTest::SetUp();
    }
};

TEST_P(Padded2ValidConvTest, CompareWithRefs) {
    Run();
}

TEST_P(Gna30Padded2ValidConvTest, CompareWithRefs) {
    Run();
}

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    //TODO: some tests fail for FP16
    //InferenceEngine::Precision::FP16
};

const std::vector<std::map<std::string, std::string>> configs1D = {
    {
        {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
        {"GNA_SCALE_FACTOR_0", "1"},
        {"GNA_EXEC_TARGET", "GNA_TARGET_2_0"}
    }
};

const std::vector<std::map<std::string, std::string>> configs1D_Gna30 = {
    {
        {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
        {"GNA_SCALE_FACTOR_0", "1"},
        {"GNA_EXEC_TARGET", "GNA_TARGET_3_0"}
    }
};

const std::vector<std::map<std::string, std::string>> configs2D = {
    {
        {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
        {"GNA_SCALE_FACTOR_0", "1"},
        {"GNA_EXEC_TARGET", "GNA_TARGET_3_0"}
    }
};

const std::vector<op::PadType> padTypes = {
        op::PadType::EXPLICIT,
        op::PadType::SAME_LOWER,
        //TODO: SAME_UPPER fails for 1d conv
        //op::PadType::SAME_UPPER,
        op::PadType::VALID
};

const std::vector<modelType> models = {
    modelType::TranspConvTransp,
    modelType::TranspConvBcastAddTransp,
    //TODO: this model fails for 1d conv
    //modelType::TranspConvBcastAddMaxPoolTransp,
    //TODO: disabled models fail with result comparison check
    //modelType::TranspConvBcastAddActTransp,
    //modelType::TranspConvBcastAddMaxPoolActTransp,
    modelType::TranspConvTranspBcastAdd,
    //modelType::TranspConvTranspBcastAddAct
};

const std::vector<std::vector<size_t>> input1DNHWC = { {1, 1, 16, 8} };
const std::vector<std::vector<size_t >> kernels1D = { {1, 2}, {1, 3} //TODO: {1, 4} fails on result comparison for 1d conv
};
const std::vector<std::vector<size_t >> strides1D = { {1, 1} };
const std::vector<std::vector<ptrdiff_t>> padBegins1D = { {0, 2} };
const std::vector<std::vector<ptrdiff_t>> padEnds1D = { {0, 3} };
const std::vector<std::vector<size_t >> dilations1D = { {1, 1} };
const std::vector<size_t> numOutChannels1D = { 4 };
const std::vector<std::vector<size_t >> biases1D = { {1, 4, 1, 1} };
const std::vector<std::vector<size_t >> transp_biases1D = { {1, 1, 1, 4} };
const std::vector<std::vector<size_t >> maxpools1D = { {1, 2} };

const std::vector<std::vector<size_t>> input2DNHWC = { {1, 16, 16, 32} };
const std::vector<std::vector<size_t >> kernels2D = { {2, 2}, {4, 1}, {1, 3} };
//TODO: strides other than {1, 1} fail on result comparison for 2d conv
const std::vector<std::vector<size_t >> strides2D = { {1, 1} };
const std::vector<std::vector<ptrdiff_t>> padBegins2D = { {1, 2} };
const std::vector<std::vector<ptrdiff_t>> padEnds2D = { {3, 1} };
const std::vector<std::vector<size_t >> dilations2D = { {1, 1} };
const std::vector<size_t> numOutChannels2D = { 32 };
const std::vector<std::vector<size_t >> biases2D = { {1, 32, 1, 1} };
const std::vector<std::vector<size_t >> transp_biases2D = { {1, 1, 1, 32} };
const std::vector<std::vector<size_t >> maxpools2D = { {2, 2} };

const auto conv1DParams = ::testing::Combine(
    ::testing::ValuesIn(kernels1D),
    ::testing::ValuesIn(strides1D),
    ::testing::ValuesIn(padBegins1D),
    ::testing::ValuesIn(padEnds1D),
    ::testing::ValuesIn(dilations1D),
    ::testing::ValuesIn(numOutChannels1D),
    ::testing::ValuesIn(padTypes),
    ::testing::ValuesIn(biases1D),
    ::testing::ValuesIn(transp_biases1D),
    ::testing::ValuesIn(maxpools1D)
);

const auto conv2DParams = ::testing::Combine(
    ::testing::ValuesIn(kernels2D),
    ::testing::ValuesIn(strides2D),
    ::testing::ValuesIn(padBegins2D),
    ::testing::ValuesIn(padEnds2D),
    ::testing::ValuesIn(dilations2D),
    ::testing::ValuesIn(numOutChannels2D),
    ::testing::ValuesIn(padTypes),
    ::testing::ValuesIn(biases2D),
    ::testing::ValuesIn(transp_biases2D),
    ::testing::ValuesIn(maxpools2D)
);

INSTANTIATE_TEST_CASE_P(smoke_1DPadded2Valid, Padded2ValidConvTest,
    ::testing::Combine(
        conv1DParams,
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(CommonTestUtils::DEVICE_GNA),
        ::testing::ValuesIn(configs1D),
        ::testing::ValuesIn(input1DNHWC),
        ::testing::ValuesIn(models)),
    Padded2ValidConvTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_1DPadded2Valid, Gna30Padded2ValidConvTest,
    ::testing::Combine(
        conv1DParams,
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(CommonTestUtils::DEVICE_GNA),
        ::testing::ValuesIn(configs1D_Gna30),
        ::testing::ValuesIn(input1DNHWC),
        ::testing::ValuesIn(models)),
    Gna30Padded2ValidConvTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_2DPadded2Valid, Gna30Padded2ValidConvTest,
    ::testing::Combine(
        conv2DParams,
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(CommonTestUtils::DEVICE_GNA),
        ::testing::ValuesIn(configs2D),
        ::testing::ValuesIn(input2DNHWC),
        ::testing::ValuesIn(models)),
    Gna30Padded2ValidConvTest::getTestCaseName);

} // namespace LayerTestsDefinitions
