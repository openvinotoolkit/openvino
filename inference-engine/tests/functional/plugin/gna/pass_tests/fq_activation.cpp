// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <memory>
#include <tuple>
#include <vector>
#include <string>

#include <ie_core.hpp>

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"

#include "ngraph_functions/pass/convert_prc.hpp"

typedef std::tuple<
    InferenceEngine::Precision,         // Network Precision
    std::string,                        // Target Device
    std::map<std::string, std::string>, // Configuration
    std::vector<size_t>,                // Input Shape
    std::pair<float, float>,            // Input Min and Max
    size_t                              // Levels
> fqActivationParams;

namespace LayerTestsDefinitions {

class FQActivation : public testing::WithParamInterface<fqActivationParams>,
    public LayerTestsUtils::LayerTestsCommon {
    float inputDataMin = 0.0f;
    float inputDataMax = 0.0f;
    float inputDataResolution = 1.0f;

public:
    static std::string getTestCaseName(testing::TestParamInfo<fqActivationParams> obj) {
        InferenceEngine::Precision netPrecision;
        std::string targetDevice;
        std::map<std::string, std::string> configuration;
        std::vector<size_t> inputShape;
        std::pair<float, float> inputMinMax;
        size_t levels = 0;
        std::tie(netPrecision, targetDevice, configuration, inputShape, inputMinMax, levels) = obj.param;

        std::ostringstream result;
        result << "netPRC=" << netPrecision.name() << "_";
        result << "targetDevice=" << targetDevice << "_";
        for (auto const& configItem : configuration) {
            result << "_configItem=" << configItem.first << "_" << configItem.second;
        }
        result << "_inputShape=" << CommonTestUtils::vec2str(inputShape);
        result << "_inputMinMax=(" << inputMinMax.first << ".." << inputMinMax.second << ")";
        result << "_levels=" << levels;

        return result.str();
    }

    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo& info) const override {
        return FuncTestUtils::createAndFillBlob(info.getTensorDesc(), inputDataMax - inputDataMin, inputDataMin, 1 / inputDataResolution);
    }

protected:
    void SetUp() override {
        InferenceEngine::Precision netPrecision;

        std::vector<size_t> inputShape;
        std::pair<float, float> inputMinMax;
        size_t levels = 0;
        std::tie(netPrecision, targetDevice, configuration, inputShape, inputMinMax, levels) = this->GetParam();
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

        auto inputLowNode = ngraph::builder::makeConstant<float>(ngPrc, { 1 }, { inputMinMax.first });
        auto inputHighNode = ngraph::builder::makeConstant<float>(ngPrc, { 1 }, { inputMinMax.second });

        auto inputVector = ngraph::builder::makeParams(ngPrc, { inputShape });
        auto inputFQNode = std::make_shared<ngraph::opset1::FakeQuantize>(inputVector[0],
            inputLowNode, inputHighNode, inputLowNode, inputHighNode, levels);

        auto relu = ngraph::builder::makeActivation(inputFQNode, ngraph::element::f32, ngraph::helpers::ActivationTypes::Relu);
        auto reluFQNode = std::make_shared<ngraph::opset1::FakeQuantize>(relu,
            inputLowNode, inputHighNode, inputLowNode, inputHighNode, levels);

        ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(reluFQNode) };
        function = std::make_shared<ngraph::Function>(results, inputVector, "FQActivation");
    }
};


TEST_P(FQActivation, CompareWithRefImpl) {
    Run();
};

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16
};

const std::vector<std::map<std::string, std::string>> configs = {
    {
        {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
    }
};

const std::vector<std::vector<size_t>> inputShape = {
    {1, 1024},
};

const std::vector<std::pair<float, float>> inputMinMax = {
    {-0.5, 0.5},
    {-2, 2},
    {-8, 8},
    {-16, 16},
    {-50, 50},
    {-100, 100},
};

const std::vector<size_t> levels = {
    65535,
};

INSTANTIATE_TEST_SUITE_P(smoke_fq_activation, FQActivation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(CommonTestUtils::DEVICE_GNA),
        ::testing::ValuesIn(configs),
        ::testing::ValuesIn(inputShape),
        ::testing::ValuesIn(inputMinMax),
        ::testing::ValuesIn(levels)),
    FQActivation::getTestCaseName);
} // namespace LayerTestsDefinitions
