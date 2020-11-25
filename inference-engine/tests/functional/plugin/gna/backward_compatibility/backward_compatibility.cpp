// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <memory>
#include <tuple>
#include <string>
#include <fstream>

#include <ie_core.hpp>

#include <shared_test_classes/base/layer_test_utils.hpp>
#include "functional_test_utils/blob_utils.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"

#ifndef BINARY_EXPORT_MODELS_PATH  // should be already defined by cmake
#define BINARY_EXPORT_MODELS_PATH ""
#endif

typedef std::tuple<
        InferenceEngine::Precision,         // Network Precision
        std::string,                        // Target Device
        std::string,                        // Name Export Model
        std::map<std::string, std::string>, // Export Configuration
        std::map<std::string, std::string>  // Import Configuration
> exportImportNetworkParams;

namespace LayerTestsDefinitions {

class BackwardCompatibilityTest : public testing::WithParamInterface<exportImportNetworkParams>,
                                  public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<exportImportNetworkParams> obj) {
        InferenceEngine::Precision netPrecision;
        std::string targetDevice;
        std::map<std::string, std::string> exportConfiguration;
        std::map<std::string, std::string> importConfiguration;
        std::string nameExportModel;
        std::tie(netPrecision, targetDevice, nameExportModel, exportConfiguration, importConfiguration) = obj.param;

        std::ostringstream result;
        result << "netPRC=" << netPrecision.name() << "_";
        result << "targetDevice=" << targetDevice << "_";
        result << "nameExportModel=" << nameExportModel << "_";
        for (auto const& configItem : exportConfiguration) {
            result << "_exportConfigItem=" << configItem.first << "_" << configItem.second;
        }
        for (auto const& configItem : importConfiguration) {
            result << "_importConfigItem=" << configItem.first << "_" << configItem.second;
        }
        return result.str();
    }

    void Run() override {
        configuration.insert(exportConfiguration.begin(), exportConfiguration.end());
        LoadNetwork();
        Infer();

        const auto& actualOutputs = GetOutputs();
        auto referenceOutputs = CalculateRefs();
        Compare(referenceOutputs, actualOutputs);

        for (auto const& configItem : importConfiguration) {
            configuration[configItem.first] = configItem.second;
        }
        std::string inputTensorBinary = BINARY_EXPORT_MODELS_PATH + nameExportModel;
        std::fstream inputStream(inputTensorBinary, std::ios_base::in | std::ios_base::binary);
        if (inputStream.fail()) {
            FAIL() << "Cannot open file to import model: " << inputTensorBinary;
        }
        auto importedNetwork = core->ImportNetwork(inputStream, targetDevice, configuration);
        auto importedOutputs = CalculateImportedNetwork(importedNetwork);
        Compare(importedOutputs, actualOutputs);
    }

protected:
    std::string test_name =
            ::testing::UnitTest::GetInstance()->current_test_info()->name();
    void SetUp() override {
        InferenceEngine::Precision netPrecision;
        std::tie(netPrecision, targetDevice, nameExportModel, exportConfiguration, importConfiguration) = this->GetParam();
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

        auto params = ngraph::builder::makeParams(ngPrc, { {1, 336} });
        auto const_eltwise = ngraph::builder::makeConstant(ngPrc, {1, 336}, std::vector<float>{-1});

        auto relu = std::make_shared<ngraph::opset1::Multiply>(params[0], const_eltwise);
        ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(relu) };
        function = std::make_shared<ngraph::Function>(results, params, "ExportBackwordCompatibility");
    }

private:
    std::map<std::string, std::string> exportConfiguration;
    std::map<std::string, std::string> importConfiguration;
    std::string nameExportModel;

    std::vector<std::vector<std::uint8_t>> CalculateImportedNetwork(InferenceEngine::ExecutableNetwork& importedNetwork) {
        auto refInferRequest = importedNetwork.CreateInferRequest();
        std::vector<InferenceEngine::InputInfo::CPtr> refInfos;
        for (const auto& input : importedNetwork.GetInputsInfo()) {
            const auto& info = input.second;
            refInfos.push_back(info);
        }

        for (std::size_t i = 0; i < inputs.size(); ++i) {
            const auto& input = inputs[i];
            const auto& info = refInfos[i];

            refInferRequest.SetBlob(info->name(), input);
        }

        refInferRequest.Infer();

        auto refOutputs = std::vector<InferenceEngine::Blob::Ptr>{};
        for (const auto& output : importedNetwork.GetOutputsInfo()) {
            const auto& name = output.first;
            refOutputs.push_back(refInferRequest.GetBlob(name));
        }

        auto referenceOutputs = std::vector<std::vector<std::uint8_t>>(refOutputs.size());
        for (std::size_t i = 0; i < refOutputs.size(); ++i) {
            const auto& reference = refOutputs[i];
            const auto refSize = reference->byteSize();

            auto& expectedOutput = referenceOutputs[i];
            expectedOutput.resize(refSize);

            auto refMemory = InferenceEngine::as<InferenceEngine::MemoryBlob>(reference);
            IE_ASSERT(refMemory);
            const auto refLockedMemory = refMemory->wmap();
            const auto referenceBuffer = refLockedMemory.as<const std::uint8_t*>();

            std::copy(referenceBuffer, referenceBuffer + refSize, expectedOutput.data());
        }

        return referenceOutputs;
    }
};

TEST_P(BackwardCompatibilityTest, CompareWithRefImpl) {
    Run();
};

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16
};

const std::vector<std::map<std::string, std::string>> exportConfigs = {
        {
                {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
                {"GNA_SCALE_FACTOR_0", "327.67"}
        }
};

const std::vector<std::map<std::string, std::string>> importConfigs = {
        {
                {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
                {"GNA_SCALE_FACTOR_0", "327.67"}
        },
};

const std::vector<std::string> nameExportModel = {"export2dot1.blob", "export2dot2.blob", "export2dot3.blob", "export2dot4.blob"};

INSTANTIATE_TEST_CASE_P(smoke_ImportOldVersion, BackwardCompatibilityTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(CommonTestUtils::DEVICE_GNA),
                                ::testing::ValuesIn(nameExportModel),
                                ::testing::ValuesIn(exportConfigs),
                                ::testing::ValuesIn(importConfigs)),
                        BackwardCompatibilityTest::getTestCaseName);

} // namespace LayerTestsDefinitions

