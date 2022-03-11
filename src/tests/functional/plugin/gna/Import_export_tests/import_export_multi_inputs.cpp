// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <memory>
#include <tuple>
#include <vector>
#include <string>
#include <fstream>

#include "ngraph_functions/builders.hpp"
#include "base/import_export_base.hpp"

namespace LayerTestsDefinitions {

class ImportMultiInput : public FuncTestUtils::ImportNetworkTestBase {
protected:
    void SetUp() override {
        std::vector<size_t> inputShape;
        InferenceEngine::Precision netPrecision;
        std::tie(inputShape, netPrecision, targetDevice, exportConfiguration, importConfiguration, applicationHeader) = this->GetParam();

        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        auto input = ngraph::builder::makeParams(ngPrc, {inputShape, inputShape});
        auto mul1 = ngraph::builder::makeEltwise(input[0], input[1], ngraph::helpers::EltwiseTypes::ADD);
        auto result = std::make_shared<ngraph::opset7::Result>(mul1);

        function = std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, input, "multiple_input");
    }
};

class ImportMultiInputChanged : public ImportMultiInput {};
class ImportMultiInputUnchanged : public ImportMultiInput {};

TEST_P(ImportMultiInputUnchanged, CompareWithRefImpl) {
    TestRun(false);
};

TEST_P(ImportMultiInputChanged, CompareWithRefImpl) {
    TestRun(true);
};

const std::vector<std::vector<size_t>> inputShape = {
    {1, 10}
};

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32
};

const std::vector<std::map<std::string, std::string>> exportConfigs = {
    {
        {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
        {"GNA_SCALE_FACTOR_0", "327.67"},
        {"GNA_SCALE_FACTOR_1", "327.67"}
    }
};

const std::vector<std::map<std::string, std::string>> importConfigsChanged = {
    {
        {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
        {"GNA_SCALE_FACTOR_0", "32767"}
    },
        {
        {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
        {"GNA_SCALE_FACTOR_1", "32767"}
    },
    {
        {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
        {"GNA_SCALE_FACTOR_0", "32767"},
        {"GNA_SCALE_FACTOR_1", "32767"}
    },
    {
        {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
        {"GNA_SCALE_FACTOR_0", "1"},
        {"GNA_SCALE_FACTOR_1", "32767"}
    }
};

const std::vector<std::map<std::string, std::string>> importConfigsUnchanged = {
    {
        {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
        {"GNA_SCALE_FACTOR_0", "327.67"}
    },
    {
        {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
        {"GNA_SCALE_FACTOR_0", "1"}
    },
    {
        {"GNA_DEVICE_MODE", "GNA_SW_EXACT"}
    },
    {
        {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
        {"GNA_SCALE_FACTOR_0", "327.67"},
        {"GNA_SCALE_FACTOR_1", "327.67"}
    },
    {
        {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
        {"GNA_SCALE_FACTOR_1", "327.67"}
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_ImportNetworkGNA, ImportMultiInputUnchanged,
                        ::testing::Combine(
                            ::testing::ValuesIn(inputShape),
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::Values(CommonTestUtils::DEVICE_GNA),
                            ::testing::ValuesIn(exportConfigs),
                            ::testing::ValuesIn(importConfigsUnchanged),
                            ::testing::Values("")),
                        ImportMultiInputUnchanged::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ImportNetworkGNA, ImportMultiInputChanged,
                        ::testing::Combine(
                            ::testing::ValuesIn(inputShape),
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::Values(CommonTestUtils::DEVICE_GNA),
                            ::testing::ValuesIn(exportConfigs),
                            ::testing::ValuesIn(importConfigsChanged),
                            ::testing::Values("")),
                        ImportMultiInputChanged::getTestCaseName);

} // namespace LayerTestsDefinitions

