// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <memory>
#include <tuple>
#include <vector>
#include <string>
#include <fstream>

#include <ie_core.hpp>

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/layer_test_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"

#include "ngraph_functions/pass/convert_prc.hpp"

typedef std::tuple<
    InferenceEngine::Precision,         // Network Precision
    std::string,                        // Target Device
    std::map<std::string, std::string>  //Configuration
> exportImportNetworkParams;

namespace LayerTestsDefinitions {

class ImportNetworkTest : public testing::WithParamInterface<exportImportNetworkParams>,
                          public LayerTestsUtils::LayerTestsCommon {
    public:
        static std::string getTestCaseName(testing::TestParamInfo<exportImportNetworkParams> obj) {
            InferenceEngine::Precision netPrecision;
            std::string targetDevice;
            std::map<std::string, std::string> configuration;
            std::tie(netPrecision, targetDevice, configuration) = obj.param;

            std::ostringstream result;
            result << "netPRC=" << netPrecision.name() << "_";
            result << "targetDevice=" << targetDevice << "_";
            for (auto const& configItem : configuration) {
                result << "_configItem=" << configItem.first << "_" << configItem.second;
            }
            return result.str();
        }

        void Run() override {
            SKIP_IF_CURRENT_TEST_IS_DISABLED()

            LoadNetwork();
            Infer();
            executableNetwork.Export("exported_model.blob");

            const auto& actualOutputs = GetOutputs();
            auto referenceOutputs = CalculateRefs();
            Compare(referenceOutputs, actualOutputs);

            std::fstream inputStream("exported_model.blob", std::ios_base::in | std::ios_base::binary);
            if (inputStream.fail()) {
                FAIL() << "Cannot open file to import model: exported_model.blob";
            }
            auto importedOutputs = CalculateImportedNetwork(inputStream);
            Compare(importedOutputs, actualOutputs);
        }

    protected:
        void SetUp() override {
            InferenceEngine::Precision netPrecision;
            std::tie(netPrecision, targetDevice, configuration) = this->GetParam();
            auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

            auto params = ngraph::builder::makeParams(ngPrc, { {1, 336} });

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
        }

    private:
        std::vector<std::vector<std::uint8_t>> CalculateImportedNetwork(std::istream& networkModel) {
            auto importedNetwork = core->ImportNetwork(networkModel, targetDevice, configuration);

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

    TEST_P(ImportNetworkTest, CompareWithRefImpl) {
        Run();
    };

    const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16
    };

    const std::vector<std::map<std::string, std::string>> configs = {
        {
            {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
            {"GNA_SCALE_FACTOR_0", "327.67"}
        }
    };

    INSTANTIATE_TEST_CASE_P(ImportNetworkCase, ImportNetworkTest,
        ::testing::Combine(
            ::testing::ValuesIn(netPrecisions),
            ::testing::Values(CommonTestUtils::DEVICE_GNA),
            ::testing::ValuesIn(configs)),
        ImportNetworkTest::getTestCaseName);

} // namespace LayerTestsDefinitions

