// Copyright (C) 2019-2020 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include <functional>

#include "ie_core.hpp"

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/layer_test_utils.hpp"


#include "single_layer_tests/pooling.hpp"

namespace LayerTestsDefinitions {

std::string PoolingLayerTest::getTestCaseName(testing::TestParamInfo<poolLayerTestParamsSet> obj) {
    poolSpecificParams poolParams;
    InferenceEngine::Precision netPrecision;
    std::vector<size_t> inputShapes;
    std::string targetDevice;
    std::tie(poolParams, netPrecision, inputShapes, targetDevice) = obj.param;
    ngraph::helpers::PoolingTypes poolType;
    std::vector<size_t> kernel, stride;
    std::vector<size_t> padBegin, padEnd;
    ngraph::op::PadType padType;
    ngraph::op::RoundingType roundingType;
    bool excludePad;
    std::tie(poolType, kernel, stride, padBegin, padEnd, roundingType, padType, excludePad) = poolParams;

    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
    switch (poolType) {
        case ngraph::helpers::PoolingTypes::MAX:
            result << "MaxPool_";
            break;
        case ngraph::helpers::PoolingTypes::AVG:
            result << "AvgPool_";
            result << "ExcludePad=" << excludePad << "_";
            break;
    }
    result << "K" << CommonTestUtils::vec2str(kernel) << "_";
    result << "S" << CommonTestUtils::vec2str(stride) << "_";
    result << "PB" << CommonTestUtils::vec2str(padBegin) << "_";
    result << "PE" << CommonTestUtils::vec2str(padEnd) << "_";
    if (padType == ngraph::op::PadType::EXPLICIT) {
        result << "Rounding=" << roundingType << "_";
    }
    result << "AutoPad=" << padType << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

std::string GlobalPoolingLayerTest::getTestCaseName(testing::TestParamInfo<globalPoolLayerTestParamsSet> obj) {
    poolSpecificParams poolParams;
    InferenceEngine::Precision netPrecision;
    std::string targetDevice;
    size_t channels;
    std::tie(poolParams, netPrecision, channels, targetDevice) = obj.param;
    ngraph::helpers::PoolingTypes poolType;
    std::vector<size_t> kernel, stride;
    std::vector<size_t> padBegin, padEnd;
    ngraph::op::PadType padType;
    ngraph::op::RoundingType roundingType;
    bool excludePad;
    std::tie(poolType, kernel, stride, padBegin, padEnd, roundingType, padType, excludePad) = poolParams;

    std::vector<size_t> inputShapes = {1, channels, kernel[0], kernel[1]};

    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
    switch (poolType) {
        case ngraph::helpers::PoolingTypes::MAX:
            result << "MaxPool_";
            break;
        case ngraph::helpers::PoolingTypes::AVG:
            result << "AvgPool_";
            result << "ExcludePad=" << excludePad << "_";
            break;
    }
    result << "K" << CommonTestUtils::vec2str(kernel) << "_";
    result << "S" << CommonTestUtils::vec2str(stride) << "_";
    result << "PB" << CommonTestUtils::vec2str(padBegin) << "_";
    result << "PE" << CommonTestUtils::vec2str(padEnd) << "_";
    if (padType == ngraph::op::PadType::EXPLICIT) {
        result << "Rounding=" << roundingType << "_";
    }
    result << "AutoPad=" << padType << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void PoolingLayerTest::SetUp() {
    poolSpecificParams poolParams;
    std::vector<size_t> inputShape;
    InferenceEngine::Precision netPrecision;
    std::tie(poolParams, netPrecision, inputShape, targetDevice) = this->GetParam();
    ngraph::helpers::PoolingTypes poolType;
    std::vector<size_t> kernel, stride;
    std::vector<size_t> padBegin, padEnd;
    ngraph::op::PadType padType;
    ngraph::op::RoundingType roundingType;
    bool excludePad;
    std::tie(poolType, kernel, stride, padBegin, padEnd, roundingType, padType, excludePad) = poolParams;

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

    std::shared_ptr<ngraph::Node> pooling = ngraph::builder::makePooling(paramOuts[0],
                                                                         stride,
                                                                         padBegin,
                                                                         padEnd,
                                                                         kernel,
                                                                         roundingType,
                                                                         padType,
                                                                         excludePad,
                                                                         poolType);

    ngraph::ResultVector results{std::make_shared<ngraph::opset3::Result>(pooling)};
    function = std::make_shared<ngraph::Function>(results, params, "pooling");
}

void GlobalPoolingLayerTest::SetUp() {
    poolSpecificParams poolParams;
    InferenceEngine::Precision netPrecision;
    size_t channels;
    std::tie(poolParams, netPrecision, channels, targetDevice) = this->GetParam();
    ngraph::helpers::PoolingTypes poolType;
    std::vector<size_t> kernel, stride;
    std::vector<size_t> padBegin, padEnd;
    ngraph::op::PadType padType;
    ngraph::op::RoundingType roundingType;
    bool excludePad;
    std::tie(poolType, kernel, stride, padBegin, padEnd, roundingType, padType, excludePad) = poolParams;

    std::vector<size_t> inputShape = {1, channels, kernel[1], kernel[0]};

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

    std::shared_ptr<ngraph::Node> pooling = ngraph::builder::makePooling(paramOuts[0],
                                                                         stride,
                                                                         padBegin,
                                                                         padEnd,
                                                                         kernel,
                                                                         roundingType,
                                                                         padType,
                                                                         excludePad,
                                                                         poolType);

    ngraph::ResultVector results{std::make_shared<ngraph::opset3::Result>(pooling)};
    function = std::make_shared<ngraph::Function>(results, params, "pooling");
}

TEST_P(PoolingLayerTest, CompareWithRefs) {
    Run();
}

TEST_P(GlobalPoolingLayerTest, CompareWithRefs) {
    Run();

    if (targetDevice == std::string{CommonTestUtils::DEVICE_GPU}) {
        PluginCache::get().reset();
    }
}
}  // namespace LayerTestsDefinitions
