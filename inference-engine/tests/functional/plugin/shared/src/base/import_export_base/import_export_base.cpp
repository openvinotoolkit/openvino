// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "base/import_export_base/import_export_base.hpp"

#include <fstream>

namespace FuncTestUtils {

std::string ImportNetworkTestBase::getTestCaseName(testing::TestParamInfo<exportImportNetworkParams> obj) {
    InferenceEngine::Precision netPrecision;
    std::string targetDevice;
    std::map<std::string, std::string> exportConfiguration;
    std::map<std::string, std::string> importConfiguration;
    std::string appHeader;
    std::tie(netPrecision, targetDevice, exportConfiguration, importConfiguration, appHeader) = obj.param;

    std::ostringstream result;
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetDevice << "_";
    for (auto const& configItem : exportConfiguration) {
        result << "_exportConfigItem=" << configItem.first << "_" << configItem.second;
    }
    for (auto const& configItem : importConfiguration) {
        result << "_importConfigItem=" << configItem.first << "_" << configItem.second;
    }
    result << "_appHeader=" << appHeader;
    return result.str();
}

void ImportNetworkTestBase::exportImportNetwork() {
    std::stringstream strm;
    strm.write(applicationHeader.c_str(), applicationHeader.size());
    executableNetwork.Export(strm);

    strm.seekg(0, strm.beg);
    std::string appHeader(applicationHeader.size(), ' ');
    strm.read(&appHeader[0], applicationHeader.size());
    ASSERT_EQ(appHeader, applicationHeader);
    executableNetwork = core->ImportNetwork(strm, targetDevice, configuration);
}

void ImportNetworkTestBase::Run() {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    configuration.insert(exportConfiguration.begin(), exportConfiguration.end());
    LoadNetwork();
    GenerateInputs();
    Infer();

    const auto& actualOutputs = GetOutputs();
    auto referenceOutputs = CalculateRefs();
    Compare(referenceOutputs, actualOutputs);

    for (auto const& configItem : importConfiguration) {
        configuration[configItem.first] = configItem.second;
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
    }
    for (const auto& next_output : importedExecNetwork.GetOutputsInfo()) {
        ASSERT_NO_THROW(compiledExecNetwork.GetOutputsInfo()[next_output.first]);
    }
    auto importedOutputs = GetOutputs();
    ASSERT_EQ(actualOutputs.size(), importedOutputs.size());
    for (size_t i = 0; i < actualOutputs.size(); i++) {
        Compare(actualOutputs[i], importedOutputs[i]);
    }
}

} // namespace FuncTestUtils
