// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_core.hpp>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "ov_models/builders.hpp"
#include "ov_models/pass/convert_prc.hpp"
#include "ov_models/utils/ov_helpers.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"

typedef std::tuple<InferenceEngine::Precision,          // Network Precision
                   std::string,                         // Target Device
                   std::map<std::string, std::string>,  // Configuration
                   std::vector<size_t>,                 // Input Shape
                   std::pair<float, float>,             // Input Min and Max
                   size_t,                              // Levels
                   size_t                               // Outputs
                   >
    fqOutputsActivationParams;

namespace LayerTestsDefinitions {

class FQOutputsActivation : public testing::WithParamInterface<fqOutputsActivationParams>,
                            public LayerTestsUtils::LayerTestsCommon {
    float inputDataMin = 0.0f;
    float inputDataMax = 0.0f;
    float inputDataResolution = 1.0f;

public:
    static std::string getTestCaseName(testing::TestParamInfo<fqOutputsActivationParams> obj) {
        InferenceEngine::Precision netPrecision;
        std::string targetDevice;
        std::map<std::string, std::string> configuration;
        std::vector<size_t> inputShape;
        std::pair<float, float> inputMinMax;
        size_t levels = 0;
        size_t outputCount = 1;
        std::tie(netPrecision, targetDevice, configuration, inputShape, inputMinMax, levels, outputCount) = obj.param;

        std::ostringstream result;
        result << "netPRC=" << netPrecision.name() << "_";
        result << "targetDevice=" << targetDevice << "_";
        for (auto const& configItem : configuration) {
            result << "_configItem=" << configItem.first << "_" << configItem.second;
        }
        result << "_inputShape=" << ov::test::utils::vec2str(inputShape);
        result << "_inputMinMax=(" << inputMinMax.first << ".." << inputMinMax.second << ")";
        result << "_levels=" << levels;
        result << "_outputs=" << outputCount;

        return result.str();
    }

    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo& info) const override {
        return FuncTestUtils::createAndFillBlob(info.getTensorDesc(),
                                                inputDataMax - inputDataMin,
                                                inputDataMin,
                                                1 / inputDataResolution);
    }

protected:
    void SetUp() override {
        InferenceEngine::Precision netPrecision;
        std::vector<size_t> inputShape;
        std::pair<float, float> inputMinMax;
        size_t levels = 0;
        size_t outputCount = 1;
        std::tie(netPrecision, targetDevice, configuration, inputShape, inputMinMax, levels, outputCount) =
            this->GetParam();
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

        auto inputLowNode = ngraph::builder::makeConstant<float>(ngPrc, {1}, {inputMinMax.first});
        auto inputHighNode = ngraph::builder::makeConstant<float>(ngPrc, {1}, {inputMinMax.second});
        ov::ParameterVector inputVector{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};
        auto split = ngraph::builder::makeSplit(inputVector[0], ngPrc, outputCount, 1);

        ngraph::ResultVector results;
        for (size_t i = 0; i < outputCount; ++i) {
            auto relu = ngraph::builder::makeActivation(split->output(i),
                                                        ngraph::element::f32,
                                                        ngraph::helpers::ActivationTypes::Sigmoid);
            auto reluFQNode = std::make_shared<ngraph::opset8::FakeQuantize>(relu,
                                                                             inputLowNode,
                                                                             inputHighNode,
                                                                             inputLowNode,
                                                                             inputHighNode,
                                                                             levels);
            results.push_back(std::make_shared<ngraph::opset8::Result>(reluFQNode));
        }
        function = std::make_shared<ngraph::Function>(results, inputVector, "FQOutputsActivation");
    }
};

TEST_P(FQOutputsActivation, CompareWithRefImpl) {
    Run();
};

const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                               InferenceEngine::Precision::FP16};

const std::vector<std::map<std::string, std::string>> configs = {{
    {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
}};

const std::vector<std::vector<size_t>> inputShape = {
    {1, 2048},
};

const std::vector<std::pair<float, float>> inputMinMax = {
    {-0.5, 0.5},
    {-16, 16},
    {-100, 100},
};

const std::vector<size_t> levels = {
    65535,
};

const std::vector<size_t> outputCount = {1, 2, 4};

INSTANTIATE_TEST_SUITE_P(smoke_fq_activation,
                         FQOutputsActivation,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs),
                                            ::testing::ValuesIn(inputShape),
                                            ::testing::ValuesIn(inputMinMax),
                                            ::testing::ValuesIn(levels),
                                            ::testing::ValuesIn(outputCount)),
                         FQOutputsActivation::getTestCaseName);
}  // namespace LayerTestsDefinitions
