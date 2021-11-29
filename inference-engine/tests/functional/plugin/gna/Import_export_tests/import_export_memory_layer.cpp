// Copyright (C) 2018-2021 Intel Corporation
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
        InferenceEngine::Precision,         // Network Precision
        std::string,                        // Target Device
        std::map<std::string, std::string>, // Export Configuration
        std::map<std::string, std::string>  // Import Configuration
> exportImportNetworkParams;

namespace LayerTestsDefinitions {

class ImportMemoryTest : public testing::WithParamInterface<exportImportNetworkParams>,
                          public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<exportImportNetworkParams> obj) {
        InferenceEngine::Precision netPrecision;
        std::string targetDevice;
        std::map<std::string, std::string> exportConfiguration;
        std::map<std::string, std::string> importConfiguration;
        std::tie(netPrecision, targetDevice, exportConfiguration, importConfiguration) = obj.param;

        std::ostringstream result;
        result << "netPRC=" << netPrecision.name() << "_";
        result << "targetDevice=" << targetDevice << "_";
        for (auto const &configItem : exportConfiguration) {
            result << "_exportConfigItem=" << configItem.first << "_" << configItem.second;
        }
        for (auto const &configItem : importConfiguration) {
            result << "_importConfigItem=" << configItem.first << "_" << configItem.second;
        }
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
        std::vector<std::string> queryToState;
        IE_SUPPRESS_DEPRECATED_START
        for (const auto &query_state : executableNetwork.QueryState()) {
            queryToState.push_back(query_state.GetName());
        }
        for (const auto &next_memory : importedNetwork.QueryState()) {
            ASSERT_TRUE(std::find(queryToState.begin(), queryToState.end(), next_memory.GetName()) != queryToState.end())
                                        << "State " << next_memory.GetName() << " expected to be in memory states but it is not!";
        }
        IE_SUPPRESS_DEPRECATED_END
        InferenceEngine::InferRequest importInfer = importedNetwork.CreateInferRequest();
        importInfer.Infer();
    }

protected:
    void SetUp() override {
        InferenceEngine::Precision netPrecision;
        std::tie(netPrecision, targetDevice, exportConfiguration, importConfiguration) = this->GetParam();
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

        auto params = ngraph::builder::makeParams(ngPrc, {{1, 336}});
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

private:
    std::map<std::string, std::string> exportConfiguration;
    std::map<std::string, std::string> importConfiguration;
};

TEST_P(ImportMemoryTest, CompareWithRefImpl) {
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
                {"GNA_SCALE_FACTOR_0", "32767"}
        },
        {
                {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
                {"GNA_SCALE_FACTOR_0", "327.67"}
        },
};

INSTANTIATE_TEST_CASE_P(smoke_ImportNetworkMemoryCase, ImportMemoryTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(CommonTestUtils::DEVICE_GNA),
                                ::testing::ValuesIn(exportConfigs),
                                ::testing::ValuesIn(importConfigs)),
                        ImportMemoryTest::getTestCaseName);

} // namespace LayerTestsDefinitions

