// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "import_export_tests/import_reshape_permute_conv.hpp"

#include <fstream>
#include <stdio.h>

using namespace LayerTestsDefinitions;

namespace {

class ImportExportGNAModelUnchanged : public ImportReshapePermuteConv {
private:
    void exportImportNetwork() override {
        {
            std::ofstream out(fileName);
            out.write(applicationHeader.c_str(), applicationHeader.size());
            executableNetwork.Export(out);
        }
        {
            std::string appHeader(applicationHeader.size(), ' ');
            std::fstream inputStream(fileName, std::ios_base::in | std::ios_base::binary);
            if (inputStream.fail()) {
                FAIL() << "Cannot open file to import model: " << fileName;
            }
            inputStream.read(&appHeader[0], applicationHeader.size());
            ASSERT_EQ(appHeader, applicationHeader);
            executableNetwork = core->ImportNetwork(inputStream, targetDevice, configuration);
        }
    }

protected:
    void TearDown() override {
        if (remove(fileName.c_str()) != 0) {
            FAIL() << "Error: could not delete file " << fileName;
        }
    }

private:
    std::string fileName = "exported_model.blob";
};

class ImportExportGNAModelChanged : public ImportExportGNAModelUnchanged {};

TEST_P(ImportExportGNAModelUnchanged, ReshapePermuteConv) {
    TestRun(false);
};

TEST_P(ImportExportGNAModelChanged, ReshapePermuteConv) {
    TestRun(true);
};

const std::vector<std::vector<size_t>> inputShapes = {
    {1, 336}
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

const std::vector<std::map<std::string, std::string>> importConfigsChanged = {
    {
        {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
        {"GNA_SCALE_FACTOR_0", "32767"}
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
    }
};

const std::vector<std::string> appHeaders = {
        "",
        "APPLICATION_HEADER"
};

INSTANTIATE_TEST_SUITE_P(smoke_ImportNetworkGNA, ImportExportGNAModelUnchanged,
                        ::testing::Combine(
                            ::testing::ValuesIn(inputShapes),
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::Values(CommonTestUtils::DEVICE_GNA),
                            ::testing::ValuesIn(exportConfigs),
                            ::testing::ValuesIn(importConfigsUnchanged),
                            ::testing::ValuesIn(appHeaders)),
                        ImportExportGNAModelUnchanged::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ImportNetworkGNA, ImportExportGNAModelChanged,
                        ::testing::Combine(
                            ::testing::ValuesIn(inputShapes),
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::Values(CommonTestUtils::DEVICE_GNA),
                            ::testing::ValuesIn(exportConfigs),
                            ::testing::ValuesIn(importConfigsChanged),
                            ::testing::ValuesIn(appHeaders)),
                        ImportExportGNAModelChanged::getTestCaseName);

} // namespace
