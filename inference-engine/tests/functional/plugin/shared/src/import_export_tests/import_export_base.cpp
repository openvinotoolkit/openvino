// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "import_export_tests/import_export_base.hpp"

#include <fstream>

namespace LayerTestsDefinitions {

std::string ImportNetworkTestBase::getTestCaseName(testing::TestParamInfo<exportImportNetworkParams> obj) {
    InferenceEngine::Precision netPrecision;
    std::string targetDevice;
    std::map<std::string, std::string> exportConfiguration;
    std::map<std::string, std::string> importConfiguration;
    std::tie(netPrecision, targetDevice, exportConfiguration, importConfiguration) = obj.param;

    std::ostringstream result;
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetDevice << "_";
    for (auto const& configItem : exportConfiguration) {
        result << "_exportConfigItem=" << configItem.first << "_" << configItem.second;
    }
    for (auto const& configItem : importConfiguration) {
        result << "_importConfigItem=" << configItem.first << "_" << configItem.second;
    }
    return result.str();
}

void ImportNetworkTestBase::Run() {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    configuration.insert(exportConfiguration.begin(), exportConfiguration.end());
    LoadNetwork();
    Infer();
    executableNetwork.Export("exported_model.blob");

    const auto& actualOutputs = GetOutputs();
    auto referenceOutputs = CalculateRefs();
    Compare(referenceOutputs, actualOutputs);

    for (auto const& configItem : importConfiguration) {
        configuration[configItem.first] = configItem.second;
    }
    std::fstream inputStream("exported_model.blob", std::ios_base::in | std::ios_base::binary);
    if (inputStream.fail()) {
        FAIL() << "Cannot open file to import model: exported_model.blob";
    }
    auto importedNetwork = core->ImportNetwork(inputStream, targetDevice, configuration);
    for (const auto& next_input : importedNetwork.GetInputsInfo()) {
        ASSERT_NO_THROW(executableNetwork.GetInputsInfo()[next_input.first]);
    }
    for (const auto& next_output : importedNetwork.GetOutputsInfo()) {
        ASSERT_NO_THROW(executableNetwork.GetOutputsInfo()[next_output.first]);
    }
    auto importedOutputs = CalculateImportedNetwork(importedNetwork);
    Compare(importedOutputs, actualOutputs);
}

std::vector<std::vector<std::uint8_t>> ImportNetworkTestBase::CalculateImportedNetwork(InferenceEngine::ExecutableNetwork& importedNetwork) {
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

} // namespace LayerTestsDefinitions
