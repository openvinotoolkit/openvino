// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
//
#include <vector>
#include <tuple>
#include <string>

#include <ie_core.hpp>

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/pass/convert_prc.hpp"

using LayersRestrictionsParamsTuple = typename std::tuple<
        InferenceEngine::Precision,         // Network Precision
        std::map<std::string, std::string>, // Configuration
        std::string>;                       // Device name

namespace LayerTestsDefinitions {

struct FullyConnectedBatchSizeMoreThan8 {
    static const char* getName() { return "FullyConnectedBatchSizeMoreThan8"; }
    static std::shared_ptr<ngraph::Function> createTopology(const InferenceEngine::Precision& netPrecision) {
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        std::vector<size_t> inputShape = {9, 1};
        auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
        auto weights = CommonTestUtils::generate_float_numbers(inputShape[1] * inputShape[1], -0.0001f, 0.0001f);
        auto fullyConnected = ngraph::builder::makeFullyConnected(params[0], ngPrc, inputShape[1], false, {}, weights);
        ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(fullyConnected) };
        return std::make_shared<ngraph::Function>(results, params, getName());
    }
    static const char* getMatch() { return "and batch size(9) not supported"; }
};

struct FullyConnectedBatchSizeLessThanOrEqual8 {
    static const char* getName() { return "FullyConnectedBatchSizeLessThanOrEqual8"; }
    static std::shared_ptr<ngraph::Function> createTopology(const InferenceEngine::Precision& netPrecision) {
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        std::vector<size_t> inputShape = {7, 1};
        auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
        auto weights = CommonTestUtils::generate_float_numbers(inputShape[1] * inputShape[1], -0.0001f, 0.0001f);
        auto fullyConnected = ngraph::builder::makeFullyConnected(params[0], ngPrc, inputShape[1], false, {}, weights);
        ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(fullyConnected) };
        return std::make_shared<ngraph::Function>(results, params, getName());
    }
};

template<typename T>
class LayersRestrictions : public testing::WithParamInterface<LayersRestrictionsParamsTuple>,
    public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<LayersRestrictionsParamsTuple> obj) {
        InferenceEngine::Precision netPrecision;
        std::map<std::string, std::string> configuration;
        std::string targetDevice;
        std::tie(netPrecision, configuration, targetDevice) = obj.param;
        std::ostringstream result;
        result << T::getName() << "_";
        result << "netPRC=" << netPrecision.name() << "_";
        result << "targetDevice=" << targetDevice << "_";
        for (auto const& configItem : configuration) {
            result << "configItem=" << configItem.first << "_" << configItem.second << "_";
        }
        return result.str();
    }
    static const char* getMatch() { return T::getMatch(); }

protected:
    void SetUp() override {
        InferenceEngine::Precision netPrecision;
        std::tie(netPrecision, configuration, targetDevice) = this->GetParam();
        function = T::createTopology(netPrecision);
    }
};

using LayersRestrictionsFullyConnectedBatchSizeMoreThan8 = LayersRestrictions<FullyConnectedBatchSizeMoreThan8>;
using LayersRestrictionsFullyConnectedBatchSizeLessThanOrEqual8 = LayersRestrictions<FullyConnectedBatchSizeLessThanOrEqual8>;

TEST_P(LayersRestrictionsFullyConnectedBatchSizeMoreThan8, CompareWithRefImpl) {
    std::string what;
    try {
        LoadNetwork();
    } catch (const std::exception& e) {
        what.assign(e.what());
    }
    EXPECT_TRUE(what.find(getMatch()) != std::string::npos);
}

TEST_P(LayersRestrictionsFullyConnectedBatchSizeLessThanOrEqual8, CompareWithRefImpl) {
    Run();
};

const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32};
const std::vector<std::map<std::string, std::string>> configs = {
    { {"GNA_DEVICE_MODE", "GNA_SW_EXACT"} }
};

INSTANTIATE_TEST_SUITE_P(smoke_layers_restrictions, LayersRestrictionsFullyConnectedBatchSizeMoreThan8,
                        ::testing::Combine(
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::ValuesIn(configs),
                                ::testing::Values(CommonTestUtils::DEVICE_GNA)),
                        LayersRestrictionsFullyConnectedBatchSizeMoreThan8::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_layers_restrictions, LayersRestrictionsFullyConnectedBatchSizeLessThanOrEqual8,
                        ::testing::Combine(
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::ValuesIn(configs),
                                ::testing::Values(CommonTestUtils::DEVICE_GNA)),
                        LayersRestrictionsFullyConnectedBatchSizeLessThanOrEqual8::getTestCaseName);
} // namespace LayerTestsDefinitions
