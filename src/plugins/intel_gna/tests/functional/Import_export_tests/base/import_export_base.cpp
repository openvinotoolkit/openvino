// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "import_export_base.hpp"
#include "../backward_compatibility/gna_plugin_factory.hpp"

#include <fstream>

using namespace ov::intel_gna::test;

namespace FuncTestUtils {

std::string ImportNetworkTestBase::getTestCaseName(testing::TestParamInfo<exportImportNetworkParams> obj) {
    std::vector<size_t> inputShape;
    InferenceEngine::Precision netPrecision;
    std::string targetDevice;
    std::map<std::string, std::string> exportConfiguration;
    std::map<std::string, std::string> importConfiguration;
    std::string appHeader;
    kExportModelVersion model_ver;

    std::tie(inputShape, netPrecision, targetDevice,
             exportConfiguration, importConfiguration,
             model_ver, appHeader) = obj.param;

    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShape) << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetDevice << "_";
    for (auto const& configItem : exportConfiguration) {
        result << "_exportConfigItem=" << configItem.first << "_" << configItem.second;
    }
    for (auto const& configItem : importConfiguration) {
        result << "_importConfigItem=" << configItem.first << "_" << configItem.second;
    }
    result << "_appHeader=" << appHeader;
    result << "_ModelVer=" << ExportModelVersionToStr(model_ver);
    return result.str();
}

void ImportNetworkTestBase::exportImportNetwork() {
    std::stringstream strm;
    strm.write(applicationHeader.c_str(), applicationHeader.size());
    if(m_model_version == kExportModelVersion::UNKNOWN) {
        executableNetwork.Export(strm);
    } else {
        auto plugin = GNAPluginLegacyFactory::CreatePluginLegacy(m_model_version);
        plugin->SetConfig(exportConfiguration);
        InferenceEngine::CNNNetwork cnn_network(function);
        plugin->LoadNetwork(cnn_network);
        plugin->Export(strm);
    }

    strm.seekg(0, strm.beg);
    std::string appHeader(applicationHeader.size(), ' ');
    strm.read(&appHeader[0], applicationHeader.size());
    ASSERT_EQ(appHeader, applicationHeader);
    executableNetwork = core->ImportNetwork(strm, targetDevice, configuration);
}

void ImportNetworkTestBase::Run() {
    functionRefs = ngraph::clone_function(*function);
    TestRun(false);
}

void ImportNetworkTestBase::TestRun(bool isModelChanged) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    functionRefs = ngraph::clone_function(*function);
    // load export configuration and save outputs
    configuration.insert(exportConfiguration.begin(), exportConfiguration.end());
    LoadNetwork();
    GenerateInputs();
    Infer();
    auto actualOutputs = GetOutputs();

    auto referenceOutputs = CalculateRefs();
    Compare(referenceOutputs, actualOutputs);

    for (auto const& configItem : importConfiguration) {
        configuration[configItem.first] = configItem.second;
    }

    // for import with different scale factor need to use import configuration to get refference outputs.
    if (isModelChanged) {
        LoadNetwork();
        GenerateInputs();
        Infer();
        actualOutputs = GetOutputs();
    }

    const auto compiledExecNetwork = executableNetwork;
    exportImportNetwork();
    const auto importedExecNetwork = executableNetwork;

    GenerateInputs();
    Infer();

    ASSERT_EQ(importedExecNetwork.GetInputsInfo().size(), compiledExecNetwork.GetInputsInfo().size());
    ASSERT_EQ(importedExecNetwork.GetOutputsInfo().size(), compiledExecNetwork.GetOutputsInfo().size());

    for (const auto& next_input : importedExecNetwork.GetInputsInfo()) {
        ASSERT_NO_THROW(compiledExecNetwork.GetInputsInfo()[next_input.first]);
        Compare(next_input.second->getTensorDesc(),
                compiledExecNetwork.GetInputsInfo()[next_input.first]->getTensorDesc());
    }
    for (const auto& next_output : importedExecNetwork.GetOutputsInfo()) {
        ASSERT_NO_THROW(compiledExecNetwork.GetOutputsInfo()[next_output.first]);
    }
    auto importedOutputs = GetOutputs();

    ASSERT_EQ(actualOutputs.size(), importedOutputs.size());

    for (size_t i = 0; i < actualOutputs.size(); i++) {
        Compare(actualOutputs[i]->getTensorDesc(), importedOutputs[i]->getTensorDesc());
        Compare(actualOutputs[i], importedOutputs[i]);
    }
}

}  // namespace FuncTestUtils
