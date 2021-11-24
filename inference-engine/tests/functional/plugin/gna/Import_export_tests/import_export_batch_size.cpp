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
    std::vector<size_t>,                // Input Shape
    InferenceEngine::Precision,         // Network Precision
    std::string,                        // Target Device
    std::map<std::string, std::string>, // Export Configuration
    std::map<std::string, std::string>  // Import Configuration
> exportImportNetworkParams;

namespace LayerTestsDefinitions {

class ImportBatchTest : public testing::WithParamInterface<exportImportNetworkParams>,
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
        result << "IS=" << CommonTestUtils::vec2str(inputShape) << "_";
        return result.str();
    }

    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo& info) const override {
        return FuncTestUtils::createAndFillBlob(info.getTensorDesc(), 0.2f, -0.1f);
    }

    void Run() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        functionRefs = ngraph::clone_function(*function);

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
        executableNetwork = importedNetwork;
        GenerateInputs();
        Infer();
        Validate();
    }

protected:
    void SetUp() override {
        InferenceEngine::Precision netPrecision;
        std::vector<size_t> inputShape;
        std::tie(inputShape, netPrecision, targetDevice, exportConfiguration, importConfiguration) = this->GetParam();
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

        auto params = ngraph::builder::makeParams(ngPrc, {inputShape});

        auto mul_const_1 = ngraph::builder::makeConstant<float>(ngPrc, { inputShape[1], 2048 },
            CommonTestUtils::generate_float_numbers(2048 * inputShape[1], -0.1f, 0.1f), false);

        auto matmul_1 = std::make_shared<ngraph::op::MatMul>(params[0], mul_const_1);
        auto sigmoid_1 = std::make_shared<ngraph::op::Sigmoid>(matmul_1);

        auto mul_const_2 = ngraph::builder::makeConstant<float>(ngPrc, { 2048, 3425 },
            CommonTestUtils::generate_float_numbers(2048 * 3425, -0.1f, 0.1f), false);

        auto matmul_2 = std::make_shared<ngraph::op::MatMul>(sigmoid_1, mul_const_2);

        function = std::make_shared<ngraph::Function>(matmul_2, params, "ExportImportNetwork");
    }

private:
    std::map<std::string, std::string> exportConfiguration;
    std::map<std::string, std::string> importConfiguration;
};

TEST_P(ImportBatchTest, CompareWithRefImpl) {
    Run();
};

const std::vector<std::vector<size_t>> inputShapes = {
    {1, 440},
    {2, 440},
    {4, 128}
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

INSTANTIATE_TEST_SUITE_P(smoke_ImportNetworkBatchCase, ImportBatchTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inputShapes),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(CommonTestUtils::DEVICE_GNA),
                                ::testing::ValuesIn(exportConfigs),
                                ::testing::ValuesIn(importConfigs)),
                        ImportBatchTest::getTestCaseName);

} // namespace LayerTestsDefinitions
