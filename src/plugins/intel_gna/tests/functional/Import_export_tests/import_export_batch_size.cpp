// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_layouts.h>

#include <fstream>
#include <ie_core.hpp>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "base/import_export_base.hpp"
#include "ov_models/builders.hpp"

namespace LayerTestsDefinitions {

class ImportBatchTest : public FuncTestUtils::ImportNetworkTestBase {
protected:
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo& info) const override {
        return FuncTestUtils::createAndFillBlob(info.getTensorDesc(), 0.2f, -0.1f);
    }

    void SetUp() override {
        InferenceEngine::Precision netPrecision;
        std::vector<size_t> inputShape;
        std::string _;
        std::tie(inputShape, netPrecision, targetDevice, exportConfiguration, importConfiguration, _) =
            this->GetParam();
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};

        auto mul_const_1 = ngraph::builder::makeConstant<float>(
            ngPrc,
            {inputShape[1], 2048},
            ov::test::utils::generate_float_numbers(2048 * inputShape[1], -0.1f, 0.1f),
            false);

        auto matmul_1 = std::make_shared<ngraph::op::MatMul>(params[0], mul_const_1);
        auto sigmoid_1 = std::make_shared<ngraph::op::Sigmoid>(matmul_1);

        auto mul_const_2 =
            ngraph::builder::makeConstant<float>(ngPrc,
                                                 {2048, 3425},
                                                 ov::test::utils::generate_float_numbers(2048 * 3425, -0.1f, 0.1f),
                                                 false);

        auto matmul_2 = std::make_shared<ngraph::op::MatMul>(sigmoid_1, mul_const_2);

        function = std::make_shared<ngraph::Function>(matmul_2, params, "ExportImportNetwork");
    }
};

TEST_P(ImportBatchTest, CompareWithRefImpl) {
    Run();
};

const std::vector<std::vector<size_t>> inputShapes = {{1, 440}, {2, 440}, {4, 128}};

const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                               InferenceEngine::Precision::FP16};

const std::vector<std::map<std::string, std::string>> exportConfigs = {
    {{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}, {"GNA_SCALE_FACTOR_0", "327.67"}}};

const std::vector<std::map<std::string, std::string>> importConfigs = {{{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}}};

const std::vector<std::string> appHeader = {""};

INSTANTIATE_TEST_SUITE_P(smoke_ImportNetworkBatchCase,
                         ImportBatchTest,
                         ::testing::Combine(::testing::ValuesIn(inputShapes),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(exportConfigs),
                                            ::testing::ValuesIn(importConfigs),
                                            ::testing::ValuesIn(appHeader)),
                         ImportBatchTest::getTestCaseName);
}  // namespace LayerTestsDefinitions
