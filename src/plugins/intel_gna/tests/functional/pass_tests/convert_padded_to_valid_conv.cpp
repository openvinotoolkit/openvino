// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <fstream>
#include <map>
#include <memory>
#include <queue>
#include <sstream>
#include <string>

#include "../shared_tests_instances/skip_tests_check.hpp"
#include "common_test_utils/test_common.hpp"
#include "ov_models/builders.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "transformations/init_node_info.hpp"

using namespace ngraph;
using namespace ngraph::opset7;

namespace LayerTestsDefinitions {

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
    TranspConvTranspBcastAdd,           /* Transpose(NHWC->NCHW) => Conv => Transpose(NCHW->NHWC) => Bias */
    TranspConvTranspBcastAddAct /* Transpose(NHWC->NCHW) => Conv => Transpose(NCHW->NHWC) => Bias => Activation Function
                                 */
};

typedef std::tuple<InferenceEngine::SizeVector,  // Kernel size
                   InferenceEngine::SizeVector,  // Strides
                   std::vector<ptrdiff_t>,       // Pad begin
                   std::vector<ptrdiff_t>,       // Pad end
                   InferenceEngine::SizeVector,  // Dilation
                   size_t,                       // Num out channels
                   op::PadType                   // Padding type
                   >
    convSpecificParams;

typedef std::tuple<InferenceEngine::SizeVector,  // Bias
                   InferenceEngine::SizeVector,  // Transposed Bias
                   InferenceEngine::SizeVector,  // Maxpool pool
                   InferenceEngine::SizeVector   // Maxpool strides
                   >
    miscSpecificParams;

typedef std::tuple<convSpecificParams,                  // Convolution parameters
                   miscSpecificParams,                  // Bias & Maxpool parameters
                   InferenceEngine::Precision,          // Network Precision
                   std::string,                         // Target Device
                   std::map<std::string, std::string>,  // Configuration
                   InferenceEngine::SizeVector,         // Input shapes
                   modelType                            // Test model
                   >
    paddedToValidParams;

class PaddedToValidConvTest : public testing::WithParamInterface<paddedToValidParams>,
                              virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<paddedToValidParams> obj) {
        convSpecificParams convParams;
        miscSpecificParams miscParams;
        InferenceEngine::Precision netPrecision;
        std::string targetDevice;
        std::map<std::string, std::string> configuration;
        InferenceEngine::SizeVector inputShape;
        modelType model;
        std::tie(convParams, miscParams, netPrecision, targetDevice, configuration, inputShape, model) = obj.param;
        op::PadType padType;
        InferenceEngine::SizeVector kernel, stride, dilation, bias, transpBias, maxpoolPool, maxpoolStride;
        std::vector<ptrdiff_t> padBegin, padEnd;
        size_t numOutChannels;
        std::tie(kernel, stride, padBegin, padEnd, dilation, numOutChannels, padType) = convParams;
        std::tie(bias, transpBias, maxpoolPool, maxpoolStride) = miscParams;

        std::ostringstream result;
        result << "M=" << static_cast<uint32_t>(model) << "_";
        result << "IS=" << ov::test::utils::vec2str(inputShape) << "_";
        result << "K" << ov::test::utils::vec2str(kernel) << "_";
        result << "S" << ov::test::utils::vec2str(stride) << "_";
        result << "PB" << ov::test::utils::vec2str(padBegin) << "_";
        result << "PE" << ov::test::utils::vec2str(padEnd) << "_";
        result << "D=" << ov::test::utils::vec2str(dilation) << "_";
        result << "O=" << numOutChannels << "_";
        result << "AP=" << padType << "_";
        result << "B=" << ov::test::utils::vec2str(bias) << "_";
        result << "B=" << ov::test::utils::vec2str(transpBias) << "_";
        result << "MPP=" << ov::test::utils::vec2str(maxpoolPool) << "_";
        result << "MPS=" << ov::test::utils::vec2str(maxpoolStride) << "_";
        result << "netPRC=" << netPrecision.name() << "_";
        result << "targetDevice=" << targetDevice << "_";
        for (auto const& configItem : configuration) {
            result << "_configItem=" << configItem.first << "_" << configItem.second;
        }
        return result.str();
    }

protected:
    GnaLayerTestCheck gnaVersionCheck;

    void SetUp() override {
        threshold = 0.015f;
        convSpecificParams convParams;
        miscSpecificParams miscParams;
        InferenceEngine::Precision netPrecision;
        std::vector<size_t> inputShape;
        modelType model;
        std::tie(convParams, miscParams, netPrecision, targetDevice, configuration, inputShape, model) =
            this->GetParam();
        op::PadType padType;
        InferenceEngine::SizeVector kernel, stride, dilation, bias, transpBias, maxpoolPool, maxpoolStride;
        std::vector<ptrdiff_t> padBegin, padEnd;
        size_t numOutChannels;
        std::tie(kernel, stride, padBegin, padEnd, dilation, numOutChannels, padType) = convParams;
        std::tie(bias, transpBias, maxpoolPool, maxpoolStride) = miscParams;
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

        Shape biasShape{bias};
        Shape transpBiasShape{transpBias};
        Shape maxpoolShape{maxpoolPool};
        Strides maxpoolStrides{maxpoolStride};

        ov::ParameterVector input{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};
        auto transposeInOrder = op::Constant::create(element::i64, Shape{4}, {0, 3, 1, 2});
        auto transposeIn = std::make_shared<Transpose>(input[0], transposeInOrder);
        auto filterSize = std::accumulate(std::begin(kernel), std::end(kernel), 1ull, std::multiplies<size_t>());
        auto filterWeights =
            ov::test::utils::generate_float_numbers(numOutChannels * inputShape[3] * filterSize, -0.05f, 0.05f);
        auto conv = builder::makeConvolution(transposeIn,
                                             ngPrc,
                                             kernel,
                                             stride,
                                             padBegin,
                                             padEnd,
                                             dilation,
                                             padType,
                                             numOutChannels,
                                             false,
                                             filterWeights);
        auto transposeOutOrder = op::Constant::create(element::i64, Shape{4}, {0, 2, 3, 1});
        auto biasWeights = ov::test::utils::generate_float_numbers(shape_size(biasShape), -1.5f, 1.5f);
        Output<Node> biasConst = std::make_shared<Constant>(ngPrc, biasShape, biasWeights);
        Output<Node> lastOp = std::make_shared<Transpose>(conv, transposeOutOrder);

        switch (model) {
        case modelType::TranspConvBcastAddTransp: {
            auto bias = std::make_shared<Add>(conv, biasConst);
            lastOp = std::make_shared<Transpose>(bias, transposeOutOrder);
        } break;

        case modelType::TranspConvActTransp: {
            auto activation = std::make_shared<Relu>(conv);
            lastOp = std::make_shared<Transpose>(activation, transposeOutOrder);
        } break;

        case modelType::TranspConvBcastAddMaxPoolTransp: {
            auto bcastAdd = std::make_shared<Add>(conv, biasConst);
            auto maxpool = std::make_shared<MaxPool>(bcastAdd,
                                                     maxpoolStrides,
                                                     Shape{0, 0},
                                                     Shape{0, 0},
                                                     maxpoolShape,
                                                     op::RoundingType::FLOOR,
                                                     op::PadType::VALID);
            auto transpose = std::make_shared<Transpose>(maxpool, transposeOutOrder);
            auto lastOp = std::make_shared<Relu>(transpose);
        } break;

        case modelType::TranspConvBcastAddActTransp: {
            auto bcastAdd = std::make_shared<Add>(conv, biasConst);
            auto activation = std::make_shared<Relu>(bcastAdd);
            lastOp = std::make_shared<Transpose>(activation, transposeOutOrder);
        } break;

        case modelType::TranspConvBcastAddMaxPoolActTransp: {
            auto bcastAdd = std::make_shared<Add>(conv, biasConst);
            auto maxpool = std::make_shared<MaxPool>(bcastAdd,
                                                     maxpoolStrides,
                                                     Shape{0, 0},
                                                     Shape{0, 0},
                                                     maxpoolShape,
                                                     op::RoundingType::FLOOR,
                                                     op::PadType::VALID);
            auto activation = std::make_shared<Relu>(maxpool);
            lastOp = std::make_shared<Transpose>(activation, transposeOutOrder);
        } break;

        case modelType::TranspConvTranspBcastAdd: {
            biasConst = std::make_shared<Constant>(ngPrc, transpBiasShape);
            lastOp = std::make_shared<Add>(lastOp, biasConst);
        } break;

        case modelType::TranspConvTranspBcastAddAct: {
            biasConst = builder::makeConstant(ngPrc, transpBiasShape, biasWeights, true);
            auto bcastAdd = std::make_shared<Add>(lastOp, biasConst);
            lastOp = std::make_shared<Relu>(bcastAdd);
        } break;

        case modelType::TranspConvTransp:
        default:
            break;
        }

        auto result = std::make_shared<Result>(lastOp);
        function = std::make_shared<Function>(ResultVector{result}, ParameterVector{input});
        gnaVersionCheck.SetUp(targetDevice);
    }
};

TEST_P(PaddedToValidConvTest, CompareWithRefs) {
    Run();
}

const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                               InferenceEngine::Precision::FP16};

const std::vector<std::map<std::string, std::string>> configs1D = {
    {{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}, {"GNA_SCALE_FACTOR_0", "1"}, {"GNA_EXEC_TARGET", "GNA_TARGET_1_0"}},
    {{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}, {"GNA_SCALE_FACTOR_0", "1"}, {"GNA_EXEC_TARGET", "GNA_TARGET_2_0"}},
    {{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}, {"GNA_SCALE_FACTOR_0", "1"}, {"GNA_EXEC_TARGET", "GNA_TARGET_3_0"}},
    {{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}, {"GNA_SCALE_FACTOR_0", "1"}, {"GNA_EXEC_TARGET", "GNA_TARGET_3_5"}}};

const std::vector<std::map<std::string, std::string>> configs2D = {
    {{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}, {"GNA_SCALE_FACTOR_0", "1"}, {"GNA_EXEC_TARGET", "GNA_TARGET_3_0"}},
    {{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}, {"GNA_SCALE_FACTOR_0", "1"}, {"GNA_EXEC_TARGET", "GNA_TARGET_3_5"}}};

const std::vector<op::PadType> padTypes = {op::PadType::VALID,
                                           op::PadType::EXPLICIT,
                                           op::PadType::SAME_LOWER,
                                           op::PadType::SAME_UPPER};

const std::vector<modelType> models = {modelType::TranspConvTransp,
                                       modelType::TranspConvBcastAddTransp,
                                       modelType::TranspConvActTransp,
                                       modelType::TranspConvBcastAddActTransp,
                                       modelType::TranspConvTranspBcastAdd,
                                       modelType::TranspConvTranspBcastAddAct,
                                       modelType::TranspConvBcastAddMaxPoolTransp,
                                       modelType::TranspConvBcastAddMaxPoolActTransp};

const std::vector<std::vector<size_t>> input1DNHWC = {{1, 1, 16, 8}};
const std::vector<std::vector<size_t>> kernels1D = {{1, 2}, {1, 3}, {1, 4}};
const std::vector<std::vector<size_t>> strides1D = {{1, 1}};
const std::vector<std::vector<ptrdiff_t>> padBegins1D = {{0, 2}};
const std::vector<std::vector<ptrdiff_t>> padEnds1D = {{0, 3}};
const std::vector<std::vector<size_t>> dilations1D = {{1, 1}};
const std::vector<size_t> numOutChannels1D = {4};
const std::vector<std::vector<size_t>> biases1D = {{1, 4, 1, 1}};
const std::vector<std::vector<size_t>> transpBiases1D = {{1, 1, 1, 4}};
const std::vector<std::vector<size_t>> maxpool1DPools = {{1, 2}};
const std::vector<std::vector<size_t>> maxpool1DStrides = {{1, 1}};

const std::vector<std::vector<size_t>> input2DNHWC = {{1, 16, 16, 32}};
const std::vector<std::vector<size_t>> kernels2D = {{2, 2}, {4, 1}};
const std::vector<std::vector<size_t>> strides2D = {{1, 1}, {2, 1}};
const std::vector<std::vector<ptrdiff_t>> padBegins2D = {{1, 2}};
const std::vector<std::vector<ptrdiff_t>> padEnds2D = {{3, 1}};
const std::vector<std::vector<size_t>> dilations2D = {{1, 1}};
const std::vector<size_t> numOutChannels2D = {8};
const std::vector<std::vector<size_t>> biases2D = {{1, 8, 1, 1}};
const std::vector<std::vector<size_t>> transpBiases2D = {{1, 1, 1, 8}};
const std::vector<std::vector<size_t>> maxpool2DPools = {{2, 2}};
const std::vector<std::vector<size_t>> maxpool2DStrides = {{2, 1}};

const auto conv1DParams = ::testing::Combine(::testing::ValuesIn(kernels1D),
                                             ::testing::ValuesIn(strides1D),
                                             ::testing::ValuesIn(padBegins1D),
                                             ::testing::ValuesIn(padEnds1D),
                                             ::testing::ValuesIn(dilations1D),
                                             ::testing::ValuesIn(numOutChannels1D),
                                             ::testing::ValuesIn(padTypes));

const auto misc1DParams = ::testing::Combine(::testing::ValuesIn(biases1D),
                                             ::testing::ValuesIn(transpBiases1D),
                                             ::testing::ValuesIn(maxpool1DPools),
                                             ::testing::ValuesIn(maxpool1DStrides));

const auto conv2DParams = ::testing::Combine(::testing::ValuesIn(kernels2D),
                                             ::testing::ValuesIn(strides2D),
                                             ::testing::ValuesIn(padBegins2D),
                                             ::testing::ValuesIn(padEnds2D),
                                             ::testing::ValuesIn(dilations2D),
                                             ::testing::ValuesIn(numOutChannels2D),
                                             ::testing::ValuesIn(padTypes));

const auto misc2DParams = ::testing::Combine(::testing::ValuesIn(biases2D),
                                             ::testing::ValuesIn(transpBiases2D),
                                             ::testing::ValuesIn(maxpool2DPools),
                                             ::testing::ValuesIn(maxpool2DStrides));

INSTANTIATE_TEST_SUITE_P(smoke_1DPaddedToValid,
                         PaddedToValidConvTest,
                         ::testing::Combine(conv1DParams,
                                            misc1DParams,
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs1D),
                                            ::testing::ValuesIn(input1DNHWC),
                                            ::testing::ValuesIn(models)),
                         PaddedToValidConvTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_2DPaddedToValid,
                         PaddedToValidConvTest,
                         ::testing::Combine(conv2DParams,
                                            misc2DParams,
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs2D),
                                            ::testing::ValuesIn(input2DNHWC),
                                            ::testing::ValuesIn(models)),
                         PaddedToValidConvTest::getTestCaseName);

}  // namespace LayerTestsDefinitions
