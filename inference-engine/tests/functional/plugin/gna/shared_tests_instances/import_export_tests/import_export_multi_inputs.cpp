// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <memory>
#include <tuple>
#include <vector>
#include <string>
#include <fstream>

#include "ngraph_functions/builders.hpp"
#include "base/import_export_base/import_export_base.hpp"

namespace LayerTestsDefinitions {

class ImportMultiInput : public FuncTestUtils::ImportNetworkTestBase {
protected:
    void SetUp() override {
        InferenceEngine::Precision netPrecision;
        std::tie(netPrecision, targetDevice, exportConfiguration, importConfiguration, applicationHeader) = this->GetParam();

        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        auto input = ngraph::builder::makeParams(ngPrc, {{1, 10}, {1, 10}});
        auto mul1 = ngraph::builder::makeEltwise(input[0], input[1], ngraph::helpers::EltwiseTypes::ADD);
        auto result = std::make_shared<ngraph::opset7::Result>(mul1);

        function = std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, input, "multiple_input");
    }
};

class ImportMultiInputFq: public FuncTestUtils::ImportNetworkTestBase {
protected:
    void SetUp() override {
        InferenceEngine::Precision netPrecision;
        std::tie(netPrecision, targetDevice, exportConfiguration, importConfiguration, applicationHeader) = this->GetParam();

        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        auto input = ngraph::builder::makeParams(ngPrc, {{1, 10}, {1, 10}});

        auto lowNodeOut = ngraph::builder::makeConstant<float>(ngPrc, {1}, { -inputDataMin * inputDataMin });
        auto highNodeOut = ngraph::builder::makeConstant<float>(ngPrc, {1}, { inputDataMax * inputDataMax });
        auto fq_input_0 = std::make_shared<ngraph::opset7::FakeQuantize>(input[0], lowNodeOut, highNodeOut,
            lowNodeOut, highNodeOut, levels);
        auto fq_input_1 = std::make_shared<ngraph::opset7::FakeQuantize>(input[1], lowNodeOut, highNodeOut,
            lowNodeOut, highNodeOut, levels);

        auto mul1 = ngraph::builder::makeEltwise(fq_input_0, fq_input_1, ngraph::helpers::EltwiseTypes::ADD);
        auto result = std::make_shared<ngraph::opset7::Result>(mul1);

        function = std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, input, "multiple_input_fq");
    }

    float inputDataMax = 0.5;
    float inputDataMin = -0.5;
    size_t levels = std::numeric_limits<uint16_t>::max();
};


class ImportMultiInputChanged : public ImportMultiInput {};
class ImportMultiInputUnchanged : public ImportMultiInput {};
class ImportMultiInputFqChanged : public ImportMultiInputFq {};
class ImportMultiInputFqUnchanged : public ImportMultiInputFq {};

TEST_P(ImportMultiInputUnchanged, CompareWithRefImpl) {
    TestRun(false);
};

TEST_P(ImportMultiInputChanged, CompareWithRefImpl) {
    TestRun(true);
};

TEST_P(ImportMultiInputFqChanged, CompareWithRefImpl) {
    TestRun(true);
};

TEST_P(ImportMultiInputFqUnchanged, CompareWithRefImpl) {
    TestRun(false);
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

const std::vector<std::map<std::string, std::string>> exportConfigsFq = {
    // TODO: this configuration is not supported by the tests,
    // because it will be replcaed by default scale factors in the core_configuration()
    // {
    //     {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
    // },
    {
        {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
        {"GNA_SCALE_FACTOR_0", "1"},
        {"GNA_SCALE_FACTOR_1", "1"}
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
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::Values(CommonTestUtils::DEVICE_GNA),
                            ::testing::ValuesIn(exportConfigs),
                            ::testing::ValuesIn(importConfigsUnchanged),
                            ::testing::Values("")),
                        ImportMultiInputUnchanged::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ImportNetworkGNA, ImportMultiInputChanged,
                        ::testing::Combine(
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::Values(CommonTestUtils::DEVICE_GNA),
                            ::testing::ValuesIn(exportConfigs),
                            ::testing::ValuesIn(importConfigsChanged),
                            ::testing::Values("")),
                        ImportMultiInputChanged::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_ImportNetworkGNA, ImportMultiInputFqChanged,
                        ::testing::Combine(
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::Values(CommonTestUtils::DEVICE_GNA),
                            ::testing::ValuesIn(exportConfigsFq),
                            ::testing::ValuesIn(importConfigsChanged),
                            ::testing::Values("")),
                        ImportMultiInputFqChanged::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_ImportNetworkGNA, ImportMultiInputFqUnchanged,
                        ::testing::Combine(
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::Values(CommonTestUtils::DEVICE_GNA),
                            ::testing::ValuesIn(exportConfigs),
                            ::testing::ValuesIn(importConfigsUnchanged),
                            ::testing::Values("")),
                        ImportMultiInputFqUnchanged::getTestCaseName);

} // namespace LayerTestsDefinitions

