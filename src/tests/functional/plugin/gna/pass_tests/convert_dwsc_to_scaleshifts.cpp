// Copyright (C) 2018-2022 Intel Corporation
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

using namespace ngraph;
using namespace ngraph::opset7;

namespace LayerTestsDefinitions {

enum class modelType {
    TranspDWSCTransp = 0,               /* Transpose(NHWC->NCHW) => DWSC (Group Convolution) => Transpose(NCHW->NHWC) */
    TranspDWSCBiasTransp,               /* Transpose(NHWC->NCHW) => DWSC => Broadcasted Add (Bias) => Transpose(NCHW->NHWC) */
};

typedef std::tuple<
    InferenceEngine::SizeVector,    // Kernel size
    InferenceEngine::SizeVector,    // Strides
    std::vector<ptrdiff_t>,         // Pad begin
    std::vector<ptrdiff_t>,         // Pad end
    InferenceEngine::SizeVector,    // Dilation
    op::PadType,                    // Padding type
    size_t,                         // Num out channels
    size_t,                         // Num groups
    InferenceEngine::SizeVector     // Bias
> DWSCParams;

typedef std::tuple<
    DWSCParams,                         // DWSC and bias parameters
    InferenceEngine::Precision,         // Network Precision
    std::string,                        // Target Device
    std::map<std::string, std::string>, // Configuration
    InferenceEngine::SizeVector,        // Input shapes
    modelType                           // Test model
> DWSCToScaleShiftsParams;

class DWSCToScaleShiftsTest : public testing::WithParamInterface<DWSCToScaleShiftsParams>,
    virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<DWSCToScaleShiftsParams> obj) {
        DWSCParams params;
        InferenceEngine::Precision netPrecision;
        std::string targetDevice;
        std::map<std::string, std::string> configuration;
        InferenceEngine::SizeVector inputShape;
        modelType model;
        std::tie(params, netPrecision, targetDevice, configuration, inputShape, model) = obj.param;
        op::PadType padType;
        InferenceEngine::SizeVector filter, stride, dilation, bias;
        std::vector<ptrdiff_t> padBegin, padEnd;
        size_t numOutChannels, numGroups;
        std::tie(filter, stride, padBegin, padEnd, dilation, padType, numOutChannels, numGroups, bias) = params;

        std::ostringstream result;
        result << "M=" << static_cast<uint32_t>(model) << "_";
        result << "IS=" << CommonTestUtils::vec2str(inputShape) << "_";
        result << "K" << CommonTestUtils::vec2str(filter) << "_";
        result << "S" << CommonTestUtils::vec2str(stride) << "_";
        result << "PB" << CommonTestUtils::vec2str(padBegin) << "_";
        result << "PE" << CommonTestUtils::vec2str(padEnd) << "_";
        result << "D=" << CommonTestUtils::vec2str(dilation) << "_";
        result << "O=" << numOutChannels << "_";
        result << "AP=" << padType << "_";
        result << "B=" << CommonTestUtils::vec2str(bias) << "_";
        result << "netPRC=" << netPrecision.name() << "_";
        result << "targetDevice=" << targetDevice << "_";
        for (auto const& configItem : configuration) {
            result << "_configItem=" << configItem.first << "_" << configItem.second;
        }
        return result.str();
    }

protected:
    void SetUp() override {
        threshold = 0.05f;
        DWSCParams params;
        InferenceEngine::Precision netPrecision;
        std::vector<size_t> inputShape;
        modelType model;
        std::tie(params, netPrecision, targetDevice, configuration, inputShape, model) = this->GetParam();
        op::PadType padType;
        InferenceEngine::SizeVector filter, stride, dilation, bias;
        std::vector<ptrdiff_t> padBegin, padEnd;
        size_t numOutChannels, numGroups;
        std::tie(filter, stride, padBegin, padEnd, dilation, padType, numOutChannels, numGroups, bias) = params;
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

        auto input = builder::makeParams(ngPrc, {inputShape});
        auto transposeInOrder = op::Constant::create(element::i64, Shape{4}, {0, 3, 1, 2});
        auto transposeIn = std::make_shared<Transpose>(input[0], transposeInOrder);
        auto filterSize = std::accumulate(std::begin(filter), std::end(filter), 1ull, std::multiplies<size_t>());
        auto filterWeights = CommonTestUtils::generate_float_numbers(numOutChannels * (inputShape[3] / numGroups) * filterSize, -0.5f, 0.5f);
        auto dwsc = builder::makeGroupConvolution(transposeIn, ngPrc, filter, stride, padBegin,
            padEnd, dilation, padType, numOutChannels, numGroups, false, filterWeights);
        auto transposeOutOrder = op::Constant::create(element::i64, Shape{4}, {0, 2, 3, 1});
        auto lastOp = std::make_shared<Transpose>(dwsc, transposeOutOrder);

        if (model == modelType::TranspDWSCBiasTransp) {
            Shape biasShape{bias};
            auto biasWeights = CommonTestUtils::generate_float_numbers(shape_size(biasShape), -1.0f, 1.0f);
            auto biasConst = std::make_shared<Constant>(ngPrc, biasShape, biasWeights);
            auto bias = std::make_shared<Add>(dwsc, biasConst);
            lastOp = std::make_shared<Transpose>(bias, transposeOutOrder);
        }

        auto result = std::make_shared<Result>(lastOp);
        function = std::make_shared<Function>(ResultVector{result}, ParameterVector{input});
    }
};

TEST_P(DWSCToScaleShiftsTest, CompareWithRefs) {
    Run();
}

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16
};

const std::vector<std::map<std::string, std::string>> configs = {
    {
        {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
        {"GNA_SCALE_FACTOR_0", "1"},
        {"GNA_PWL_UNIFORM_DESIGN", "NO"}
    },
    {
        {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
        {"GNA_SCALE_FACTOR_0", "1"},
        {"GNA_PWL_UNIFORM_DESIGN", "YES"}
    }
};

const std::vector<op::PadType> padTypes = {
        op::PadType::VALID,
        op::PadType::EXPLICIT,
        op::PadType::SAME_LOWER,
        op::PadType::SAME_UPPER
};

const std::vector<modelType> models = {
    modelType::TranspDWSCTransp,
    modelType::TranspDWSCBiasTransp
};

const std::vector<std::vector<size_t>> inputNHWC = {{1, 1, 5, 32}};
const std::vector<std::vector<size_t >> filters = {{1, 3}};
const std::vector<std::vector<size_t >> strides = {{1, 1}, {1, 2}};
const std::vector<std::vector<ptrdiff_t>> padBegins = {{0, 1}, {0, 2}};
const std::vector<std::vector<ptrdiff_t>> padEnds = {{0, 1}};
const std::vector<std::vector<size_t >> dilations = {{1, 1}};
const std::vector<size_t> numOutChannels = {32};
const std::vector<size_t> numGroups = {32};
const std::vector<std::vector<size_t >> biases = {{1, 32, 1, 1}};

const auto convParams = ::testing::Combine(
    ::testing::ValuesIn(filters),
    ::testing::ValuesIn(strides),
    ::testing::ValuesIn(padBegins),
    ::testing::ValuesIn(padEnds),
    ::testing::ValuesIn(dilations),
    ::testing::ValuesIn(padTypes),
    ::testing::ValuesIn(numOutChannels),
    ::testing::ValuesIn(numGroups),
    ::testing::ValuesIn(biases)
);

INSTANTIATE_TEST_SUITE_P(smoke_DWSCToScaleShifts, DWSCToScaleShiftsTest,
    ::testing::Combine(
        convParams,
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(CommonTestUtils::DEVICE_GNA),
        ::testing::ValuesIn(configs),
        ::testing::ValuesIn(inputNHWC),
        ::testing::ValuesIn(models)),
    DWSCToScaleShiftsTest::getTestCaseName);

/* ============= Strides & Dilations Combination ============= */

const std::vector<op::PadType> padTypesSD = {
    op::PadType::VALID,
};

const std::vector<std::vector<size_t>> inputNHWCSD = {{1, 1, 8, 32}};
const std::vector<std::vector<size_t >> dilationsSD = {{1, 1}, {1, 2}};

const auto convParamsSD = ::testing::Combine(
    ::testing::ValuesIn(filters),
    ::testing::ValuesIn(strides),
    ::testing::ValuesIn(padBegins),
    ::testing::ValuesIn(padEnds),
    ::testing::ValuesIn(dilationsSD),
    ::testing::ValuesIn(padTypesSD),
    ::testing::ValuesIn(numOutChannels),
    ::testing::ValuesIn(numGroups),
    ::testing::ValuesIn(biases)
);

INSTANTIATE_TEST_SUITE_P(smoke_DWSCToScaleShiftsStridesDilations, DWSCToScaleShiftsTest,
    ::testing::Combine(
        convParamsSD,
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(CommonTestUtils::DEVICE_GNA),
        ::testing::ValuesIn(configs),
        ::testing::ValuesIn(inputNHWCSD),
        ::testing::ValuesIn(models)),
    DWSCToScaleShiftsTest::getTestCaseName);

} // namespace LayerTestsDefinitions
