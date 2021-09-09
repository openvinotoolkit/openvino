// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <functional_test_utils/precision_utils.hpp>
#include <ie_core.hpp>
#include <ngraph_functions/builders.hpp>
#include <test_model_repo.hpp>
#include <single_layer_common.hpp>
#include "gtest/gtest.h"

//TODO : need move to new test infrastructure @IrinaEfode
using namespace InferenceEngine;

typedef std::tuple<
        InferenceEngine::Precision,         // Network Precision
        std::string,                        // Target Device
        std::string,                        // Name Export Model
        std::map<std::string, std::string>, // Export Configuration
        std::map<std::string, std::string>  // Import Configuration
> exportImportNetworkParams;

class BackwardCompatibilityTests : public testing::WithParamInterface<exportImportNetworkParams>,
       public testing::Test{
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

    void Run() {
        InferenceEngine::Precision netPrecision;
        std::string targetDevice;
        std::map<std::string, std::string> exportConfiguration;
        std::map<std::string, std::string> importConfiguration;
        std::string nameExportModel;
        std::tie(netPrecision, targetDevice, nameExportModel, exportConfiguration, importConfiguration) = this->GetParam();
        GenerateFunction();
        Core ie;
        CNNNetwork network = CNNNetwork(function);
        ExecutableNetwork executableNetwork = ie.LoadNetwork(network, "GNA", exportConfiguration);
        InferRequest inferRequest = executableNetwork.CreateInferRequest();
        inferRequest.Infer();
        auto refOutputs = std::vector<InferenceEngine::Blob::Ptr>{};
        for (const auto& output : executableNetwork.GetOutputsInfo()) {
            const auto& name = output.first;
            refOutputs.push_back(inferRequest.GetBlob(name));
        }

        auto models = TestDataHelpers::get_data_path() + "/gna/" + nameExportModel;
        auto ImportNetwork = ie.ImportNetwork(models, "GNA", importConfiguration);
        InferRequest inferRequestImport = ImportNetwork.CreateInferRequest();
        auto input_names = executableNetwork.GetInputsInfo();
        for (const auto& input_name : input_names) {
            auto i_blob = inferRequest.GetBlob(input_name.first);
            for (const auto& infer_name : ImportNetwork.GetInputsInfo()) {
                inferRequestImport.SetBlob(infer_name.first, i_blob);
            }
        }
        inferRequestImport.Infer();
        for (const auto& output : ImportNetwork.GetOutputsInfo()) {
            const auto& name = output.first;
            refOutputs.push_back(inferRequestImport.GetBlob(name));
        }
        CompareCommonExact(refOutputs[1], refOutputs[0]);
    }

protected:
    void SetUp() override {
    }
private:
    std::shared_ptr<ngraph::Function> function;
    void GenerateFunction() {
        auto param = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 336});
        auto const_eltwise = std::make_shared<ngraph::opset1::Constant>(ngraph::element::f32, ngraph::Shape{1, 336},
                std::vector<float>{-1});
        auto relu = std::make_shared<ngraph::opset1::Multiply>(param, const_eltwise);
        ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(relu) };
        function = std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{param}, "ExportBackwordCompatibility");
    }
};

TEST_P(BackwardCompatibilityTests, smoke_BackwardCompatibility){
    Run();
}

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

INSTANTIATE_TEST_SUITE_P(smoke_OldVersion, BackwardCompatibilityTests,
                        ::testing::Combine(
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values("GNA"),
                                ::testing::ValuesIn(nameExportModel),
                                ::testing::ValuesIn(exportConfigs),
                                ::testing::ValuesIn(importConfigs)),
                        BackwardCompatibilityTests::getTestCaseName);