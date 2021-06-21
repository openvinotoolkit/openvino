// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <memory>
#include <tuple>
#include <vector>
#include <string>
#include <fstream>

#include <ie_core.hpp>
#include <ie_layouts.h>
#include "base/import_export_base/import_export_base.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"

namespace LayerTestsDefinitions {

class ImportActConvActTest : public FuncTestUtils::ImportNetworkTestBase {
protected:
    void SetUp() override {
        std::vector<size_t> inputShape;
        InferenceEngine::Precision netPrecision;
        std::tie(inputShape, netPrecision, targetDevice, exportConfiguration, importConfiguration, applicationHeader) = this->GetParam();

        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
        auto relu1 = std::make_shared<ngraph::opset1::Relu>(params[0]);

        size_t num_out_channels = 8;
        size_t kernel_size = 8;
        std::vector<float> filter_weights = CommonTestUtils::generate_float_numbers(num_out_channels * inputShape[1] * kernel_size,
                                                                                    -0.2f, 0.2f);
        auto conv = ngraph::builder::makeConvolution(relu1, ngPrc, { 1, kernel_size }, { 1, 1 }, { 0, 0 }, { 0, 0 }, { 1, 1 },
                                                     ngraph::op::PadType::VALID, num_out_channels, true, filter_weights);

        auto relu2 = std::make_shared<ngraph::opset1::Relu>(conv);
        ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(relu2)};
        function = std::make_shared<ngraph::Function>(results, params, "ExportImportNetwork");
    }

private:
    std::map<std::string, std::string> exportConfiguration;
    std::map<std::string, std::string> importConfiguration;
};

TEST_P(ImportActConvActTest, CompareWithRefImpl) {
    Run();
};

const std::vector<std::vector<size_t>> inputShape = {
    {1, 1, 1, 240},
    {1, 1, 1, 160},
    {1, 2, 1, 80}
};

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16
};

const std::vector<std::map<std::string, std::string>> exportConfigs = {
        {
                {"GNA_DEVICE_MODE", "GNA_SW_EXACT"}
        }
};

const std::vector<std::map<std::string, std::string>> importConfigs = {
        {
                {"GNA_DEVICE_MODE", "GNA_SW_EXACT"}
        }
};

const std::vector<std::string> appHeaders = {
        ""
};

INSTANTIATE_TEST_CASE_P(smoke_ImportActConvAct, ImportActConvActTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inputShape),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(CommonTestUtils::DEVICE_GNA),
                                ::testing::ValuesIn(exportConfigs),
                                ::testing::ValuesIn(importConfigs),
                                ::testing::ValuesIn(appHeaders)),
                        ImportActConvActTest::getTestCaseName);

} // namespace LayerTestsDefinitions

