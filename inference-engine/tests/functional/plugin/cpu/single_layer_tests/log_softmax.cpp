// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils/cpu_test_utils.hpp"
#include "ngraph_functions/builders.hpp"

using namespace ngraph;
using namespace InferenceEngine;
using namespace CPUTestUtils;

namespace CPULayerTestsDefinitions {

using inputShapesPair = std::pair<std::vector<ov::PartialShape>, std::vector<std::vector<ov::Shape>>>;

using logSoftmaxLayerTestParams = std::tuple<
        inputShapesPair,                       // inputShape
        InferenceEngine::Precision,            // netPrecision
        int64_t,                               // axis
        std::map<std::string, std::string>>;   // config

class LogSoftmaxLayerCPUTest : public testing::WithParamInterface<logSoftmaxLayerTestParams>, public LayerTestsUtils::LayerTestsCommon, public CPUTestsBase {
public:
static std::string getTestCaseName(testing::TestParamInfo<logSoftmaxLayerTestParams> obj) {
    inputShapesPair inputShapes;
    Precision netPrecision;
    int64_t axis;
    std::map<std::string, std::string> config;
    std::tie(inputShapes, netPrecision, axis, config) = obj.param;

    std::ostringstream result;
    result << "IS=" << CommonTestUtils::partialShape2str(inputShapes.first) << "_";
    result << "TS=";
    for (const auto& shape : inputShapes.second) {
        result << "(";
        for (const auto& item : shape) {
            result << CommonTestUtils::vec2str(item) << "_";
        }
        result << ")_";
    }
    result << "netPRC=" << netPrecision.name();
    result << "Axis=" << axis;
    return result.str();
}

protected:
void SetUp() override {
    targetDevice = CommonTestUtils::DEVICE_CPU;

    inputShapesPair inputShapes;
    Precision netPrecision;
    int64_t axis;
    std::map<std::string, std::string> config;
    std::tie(inputShapes, netPrecision, axis, config) = this->GetParam();
    configuration.insert(config.begin(), config.end());
    inPrc = outPrc = netPrecision;
    selectedType = std::string("unknown_") + inPrc.name();

    targetStaticShapes.reserve(inputShapes.second.size());
    for (size_t i = 0; i < inputShapes.second.size(); i++) {
        targetStaticShapes.push_back(inputShapes.second[i]);
    }
    inputDynamicShapes = inputShapes.first;

    const auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    const auto params = ngraph::builder::makeParams(ngPrc, {targetStaticShapes.front().front()});
    const auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
    const auto logSoftmax = std::make_shared<ngraph::op::v5::LogSoftmax>(paramOuts.at(0), axis);
    const ngraph::ResultVector results {std::make_shared<ngraph::opset1::Result>(logSoftmax)};
    function = std::make_shared<ngraph::Function>(results, params, "logSoftmax");
}
};

TEST_P(LogSoftmaxLayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
    CheckPluginRelatedResults(executableNetwork, "logSoftmax");
}

namespace {
std::map<std::string, std::string> config {};

const std::vector<InferenceEngine::Precision> netPrecisions = {
        Precision::FP32
};

const std::vector<inputShapesPair> inputShapes2D = {
        {
                // dynamic
                {
                        {-1, -1}
                },
                // target
                {
                        {{1, 100}},
                        {{100, 1}},
                        {{10, 10}}
                }
        },
        {
                // dynamic
                {
                        {-1, {1}}
                },
                // target
                {
                        {{1, 1}},
                        {{100, 1}},
                        {{10, 1}}
                }
        }
};

const std::vector<int64_t> axis2D = {
        -2, -1, 0, 1
};

const auto params2D = testing::Combine(
        testing::ValuesIn(inputShapes2D),
        testing::ValuesIn(netPrecisions),
        testing::ValuesIn(axis2D),
        testing::Values(config));

INSTANTIATE_TEST_SUITE_P(smoke_LogSoftmax2D_dynamic, LogSoftmaxLayerCPUTest, params2D, LogSoftmaxLayerCPUTest::getTestCaseName);

const std::vector<inputShapesPair> inputShapes4D = {
        {
                // dynamic
                {
                        {-1, -1, -1, -1}
                },
                // target
                {
                        {{1, 100, 1, 1}},
                        {{1, 3, 4, 3}},
                        {{2, 3, 4, 5}}
                }
        },
        {
                // dynamic
                {
                        {{1, 2}, -1, {1, 5}, -1}
                },
                // target
                {
                        {{1, 100, 1, 1}},
                        {{1, 3, 5, 3}},
                        {{2, 3, 4, 5}}
                }
        }
};

const std::vector<int64_t> axis4D = {
        -4, -3, -2, -1, 0, 1, 2, 3
};

const auto params4D = testing::Combine(
        testing::ValuesIn(inputShapes4D),
        testing::ValuesIn(netPrecisions),
        testing::ValuesIn(axis4D),
        testing::Values(config));

INSTANTIATE_TEST_SUITE_P(smoke_LogSoftmax4D_dynamic, LogSoftmaxLayerCPUTest, params4D, LogSoftmaxLayerCPUTest::getTestCaseName);

} // namespace

} // namespace CPULayerTestsDefinitions
