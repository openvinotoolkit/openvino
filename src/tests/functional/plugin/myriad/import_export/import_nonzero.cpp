// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/opsets/opset5.hpp"

#include "base/import_export_base/import_export_base.hpp"

namespace LayerTestsDefinitions {

class ImportNonZero : public FuncTestUtils::ImportNetworkTestBase {
protected:
    void SetUp() override {
        InferenceEngine::Precision netPrecision;
        ngraph::Shape inputShape;
        std::tie(inputShape, netPrecision, targetDevice, exportConfiguration, importConfiguration, applicationHeader) = this->GetParam();
        const auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

        const auto parameter = std::make_shared<ngraph::opset5::Parameter>(ngPrc, inputShape);
        const auto nonZero = std::make_shared<ngraph::opset5::NonZero>(parameter);

        function = std::make_shared<ngraph::Function>(nonZero->outputs(), ngraph::ParameterVector{parameter}, "ExportImportNetwork");
    }
};

TEST_P(ImportNonZero, CompareWithRefImpl) {
    Run();
};

} // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
};

const std::vector<std::map<std::string, std::string>> exportConfigs = {
        {}
};

const std::vector<std::map<std::string, std::string>> importConfigs = {
        {}
};

const std::vector<std::string> appHeaders = {
        "",
        "APPLICATION_HEADER"
};

std::vector<size_t> inputShape = ngraph::Shape{1000};

INSTANTIATE_TEST_SUITE_P(smoke_ImportNetworkCase, ImportNonZero,
                         ::testing::Combine(
                                 ::testing::Values(inputShape),
                                 ::testing::ValuesIn(netPrecisions),
                                 ::testing::Values(CommonTestUtils::DEVICE_MYRIAD),
                                 ::testing::ValuesIn(exportConfigs),
                                 ::testing::ValuesIn(importConfigs),
                                 ::testing::ValuesIn(appHeaders)),
                         ImportNonZero::getTestCaseName);

} // namespace
