// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "base/import_export_base.hpp"
#include "ngraph_functions/builders.hpp"

namespace LayerTestsDefinitions {

class ImportReshapePermuteConv : public FuncTestUtils::ImportNetworkTestBase {
protected:
    void SetUp() override {
        std::vector<size_t> inputShape;
        InferenceEngine::Precision netPrecision;
        std::tie(inputShape, netPrecision, targetDevice, exportConfiguration, importConfiguration, applicationHeader) = this->GetParam();
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

        auto params = ngraph::builder::makeParams(ngPrc, { inputShape });

        std::vector<size_t> outFormShapes1 = { 1, 1, 168, 2 };
        auto pattern1 = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{ 4 }, outFormShapes1);
        auto reshape1 = std::make_shared<ngraph::opset1::Reshape>(params[0], pattern1, false);

        auto permute1 = std::make_shared<ngraph::opset1::Transpose>(reshape1,
                                                                    ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{ 4 }, { 0, 3, 1, 2 }));

        auto conv1 = ngraph::builder::makeConvolution(permute1, ngPrc, { 1, 8 }, { 1, 1 }, { 0, 0 }, { 0, 0 }, { 1, 1 },
                                                      ngraph::op::PadType::VALID, 12);

        auto permute2 = std::make_shared<ngraph::opset1::Transpose>(conv1,
                                                                    ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{ 4 }, { 0, 2, 3, 1 }));

        std::vector<size_t> outFormShapes2 = { 1, 1932 };
        auto pattern2 = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{ 2 }, outFormShapes2);
        auto reshape2 = std::make_shared<ngraph::opset1::Reshape>(permute2, pattern2, false);

        ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(reshape2) };
        function = std::make_shared<ngraph::Function>(results, params, "ExportImportNetwork");
    };
};

TEST_P(ImportReshapePermuteConv, CompareWithRefImpl) {
    Run();
};

} // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

class ImportExportGNAModelUnchanged : public ImportReshapePermuteConv {
private:
    void exportImportNetwork() override {
        {
            std::fstream outStream(fileName, std::ios_base::out | std::ios_base::binary);
            outStream.write(applicationHeader.c_str(), applicationHeader.size());
            executableNetwork.Export(outStream);
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

