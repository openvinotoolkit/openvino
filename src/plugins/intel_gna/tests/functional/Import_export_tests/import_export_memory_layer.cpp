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

#include "functional_test_utils/blob_utils.hpp"
#include "ov_models/builders.hpp"
#include "ov_models/utils/ov_helpers.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"

typedef std::tuple<InferenceEngine::Precision,          // Network Precision
                   std::string,                         // Target Device
                   std::map<std::string, std::string>,  // Export Configuration
                   std::map<std::string, std::string>,  // Import Configuration
                   std::pair<bool, bool>                // With reset
                   >
    exportImportNetworkParams;

namespace LayerTestsDefinitions {

class ImportMemoryTest : public testing::WithParamInterface<exportImportNetworkParams>,
                         public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<exportImportNetworkParams> obj) {
        InferenceEngine::Precision netPrecision;
        std::string targetDevice;
        std::map<std::string, std::string> exportConfiguration;
        std::map<std::string, std::string> importConfiguration;
        std::pair<bool, bool> withReset;
        std::tie(netPrecision, targetDevice, exportConfiguration, importConfiguration, withReset) = obj.param;

        std::ostringstream result;
        result << "netPRC=" << netPrecision.name() << "_";
        result << "targetDevice=" << targetDevice << "_";
        for (auto const& configItem : exportConfiguration) {
            result << "_exportConfigItem=" << configItem.first << "_" << configItem.second;
        }
        for (auto const& configItem : importConfiguration) {
            result << "_importConfigItem=" << configItem.first << "_" << configItem.second;
        }
        result << "_resetBefore=" << withReset.first;
        result << "_resetAfter=" << withReset.second;
        return result.str();
    }

    void Run() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED()

        configuration.insert(exportConfiguration.begin(), exportConfiguration.end());
        LoadNetwork();
        GenerateInputs();
        Infer();
        if (withReset.first) {
            for (auto& query_state : inferRequest.QueryState()) {
                query_state.Reset();
            }
        }
        executableNetwork.Export("exported_model.blob");
        for (auto const& configItem : importConfiguration) {
            configuration[configItem.first] = configItem.second;
        }
        std::fstream inputStream("exported_model.blob", std::ios_base::in | std::ios_base::binary);
        if (inputStream.fail()) {
            FAIL() << "Cannot open file to import model: exported_model.blob";
        }
        auto importedNetwork = core->ImportNetwork(inputStream, targetDevice, configuration);
        std::vector<std::string> queryToState;
        InferenceEngine::InferRequest importInfer = importedNetwork.CreateInferRequest();

        for (auto& query_state : importInfer.QueryState()) {
            queryToState.push_back(query_state.GetName());
        }
        if (withReset.first) {
            CheckQueryStates(&inferRequest);
        }
        for (const auto& next_memory : importInfer.QueryState()) {
            ASSERT_TRUE(std::find(queryToState.begin(), queryToState.end(), next_memory.GetName()) !=
                        queryToState.end())
                << "State " << next_memory.GetName() << " expected to be in memory states but it is not!";
        }
        importInfer.Infer();
        if (withReset.second) {
            for (auto& query_state : importInfer.QueryState()) {
                query_state.Reset();
            }
            CheckQueryStates(&importInfer);
        }
    }

protected:
    void SetUp() override {
        InferenceEngine::Precision netPrecision;
        std::tie(netPrecision, targetDevice, exportConfiguration, importConfiguration, withReset) = this->GetParam();
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape{1, 336})};
        auto mem_c = ngraph::builder::makeConstant(ngPrc, {1, 336}, std::vector<size_t>{1});
        auto mem_r = std::make_shared<ngraph::opset3::ReadValue>(mem_c, "id");

        auto mul = std::make_shared<ngraph::opset1::Multiply>(params[0], mem_r);
        auto mem_w = std::make_shared<ngraph::opset3::Assign>(mul, "id");

        auto relu = std::make_shared<ngraph::opset1::Relu>(mul);
        mem_w->add_control_dependency(mem_r);
        relu->add_control_dependency(mem_w);
        ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(relu)};
        function = std::make_shared<ngraph::Function>(results, params, "ExportImportNetwork");
    }

    void CheckQueryStates(InferenceEngine::InferRequest* inferRequest) {
        for (auto& query_state : inferRequest->QueryState()) {
            auto state = query_state.GetState();
            auto state_data = state->cbuffer().as<int16_t*>();
            for (int i = 0; i < state->size(); i++) {
                EXPECT_NEAR(0, state_data[i], 1e-5);
            }
        }
    }

private:
    std::pair<bool, bool> withReset;
    std::map<std::string, std::string> exportConfiguration;
    std::map<std::string, std::string> importConfiguration;
};

TEST_P(ImportMemoryTest, CompareWithRefImpl) {
    Run();
};

const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                               InferenceEngine::Precision::FP16};

const std::vector<std::map<std::string, std::string>> exportConfigs = {
    {{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}, {"GNA_SCALE_FACTOR_0", "327.67"}}};

const std::vector<std::map<std::string, std::string>> importConfigs = {
    {{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}, {"GNA_SCALE_FACTOR_0", "32767"}},
    {{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}, {"GNA_SCALE_FACTOR_0", "327.67"}},
};

const std::vector<std::pair<bool, bool>> withReset = {
    {false, false},
    {true, false},  // Reset before export
    {false, true}   // Reset after export
};

INSTANTIATE_TEST_SUITE_P(smoke_ImportNetworkMemoryCase,
                         ImportMemoryTest,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(exportConfigs),
                                            ::testing::ValuesIn(importConfigs),
                                            ::testing::ValuesIn(withReset)),
                         ImportMemoryTest::getTestCaseName);

}  // namespace LayerTestsDefinitions
