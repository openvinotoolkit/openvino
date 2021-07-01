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

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"

typedef std::tuple<
        std::vector<size_t>,                // Input shape
        InferenceEngine::Precision,         // Network Precision
        std::string,                        // Target Device
        std::map<std::string, std::string>, // Export Configuration
        std::map<std::string, std::string>  // Import Configuration
> exportImportNetworkParams;

namespace LayerTestsDefinitions {

class ImportActConvActTest : public testing::WithParamInterface<exportImportNetworkParams>,
                                 public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<exportImportNetworkParams> obj) {
        std::vector<size_t> inputShape;
        InferenceEngine::Precision netPrecision;
        std::string targetDevice;
        std::map<std::string, std::string> exportConfiguration;
        std::map<std::string, std::string> importConfiguration;
        std::tie(inputShape, netPrecision, targetDevice, exportConfiguration, importConfiguration) = obj.param;

        std::ostringstream result;
        result << "netPRC=" << netPrecision.name() << "_";
        result << "targetDevice=" << targetDevice << "_";
        for (auto const &configItem : exportConfiguration) {
            result << "_exportConfigItem=" << configItem.first << "_" << configItem.second;
        }
        for (auto const &configItem : importConfiguration) {
            result << "_importConfigItem=" << configItem.first << "_" << configItem.second;
        }
        result << CommonTestUtils::vec2str(inputShape);
        return result.str();
    }

    void Run() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED()

        configuration.insert(exportConfiguration.begin(), exportConfiguration.end());
        LoadNetwork();
        GenerateInputs();
        Infer();

        executableNetwork.Export("exported_model.blob");
        for (auto const &configItem : importConfiguration) {
            configuration[configItem.first] = configItem.second;
        }
        std::fstream inputStream("exported_model.blob", std::ios_base::in | std::ios_base::binary);
        if (inputStream.fail()) {
            FAIL() << "Cannot open file to import model: exported_model.blob";
        }

        auto importedNetwork = core->ImportNetwork(inputStream, targetDevice, configuration);

        // Generate inputs
        std::vector<InferenceEngine::Blob::Ptr> inputs;
        auto inputsInfo = importedNetwork.GetInputsInfo();
        auto functionParams = function->get_parameters();
        for (int i = 0; i < functionParams.size(); ++i) {
            const auto& param = functionParams[i];
            const auto infoIt = inputsInfo.find(param->get_friendly_name());
            GTEST_ASSERT_NE(infoIt, inputsInfo.cend());

            const auto& info = infoIt->second;
            auto blob = GenerateInput(*info);
            inputs.push_back(blob);
        }

        // Infer imported network
        InferenceEngine::InferRequest importInfer = importedNetwork.CreateInferRequest();
        inputsInfo = importedNetwork.GetInputsInfo();
        functionParams = function->get_parameters();
        for (int i = 0; i < functionParams.size(); ++i) {
            const auto& param = functionParams[i];
            const auto infoIt = inputsInfo.find(param->get_friendly_name());
            GTEST_ASSERT_NE(infoIt, inputsInfo.cend());

            const auto& info = infoIt->second;
            auto blob = inputs[i];
            importInfer.SetBlob(info->name(), blob);
        }
        importInfer.Infer();

        // Validate
        auto expectedOutputs = CalculateRefs();
        auto actualOutputs = std::vector<InferenceEngine::Blob::Ptr>{};
        for (const auto &output : importedNetwork.GetOutputsInfo()) {
            const auto &name = output.first;
            actualOutputs.push_back(importInfer.GetBlob(name));
        }
        IE_ASSERT(actualOutputs.size() == expectedOutputs.size())
        << "nGraph interpreter has " << expectedOutputs.size() << " outputs, while IE " << actualOutputs.size();
        Compare(expectedOutputs, actualOutputs);
    }

protected:
    void SetUp() override {
        std::vector<size_t> inputShape;
        InferenceEngine::Precision netPrecision;
        std::tie(inputShape, netPrecision, targetDevice, exportConfiguration, importConfiguration) = this->GetParam();
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

INSTANTIATE_TEST_SUITE_P(smoke_ImportActConvAct, ImportActConvActTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inputShape),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(CommonTestUtils::DEVICE_GNA),
                                ::testing::ValuesIn(exportConfigs),
                                ::testing::ValuesIn(importConfigs)),
                        ImportActConvActTest::getTestCaseName);

} // namespace LayerTestsDefinitions

