// Copyright (C) 2021 Intel Corporation
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
    std::pair<float, float>,            // Min, max values of weights
    size_t                              // Levels
> fqFusionWithMultipleWeightsParams;

namespace LayerTestsDefinitions {

class FQFusionWithMultipleWeights : public testing::WithParamInterface<fqFusionWithMultipleWeightsParams>,
    public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<fqFusionWithMultipleWeightsParams> obj) {
        InferenceEngine::Precision netPrecision;
        std::string targetDevice;
        std::map<std::string, std::string> configuration;
        std::vector<size_t> inputShape;
        std::pair<float, float> weightsMinMax;
        size_t levels;
        std::tie(netPrecision, targetDevice, configuration, inputShape, weightsMinMax, levels) = obj.param;

        std::ostringstream result;
        result << "netPRC=" << netPrecision.name() << "_";
        result << "targetDevice=" << targetDevice << "_";
        for (auto const& configItem : configuration) {
            result << "_configItem=" << configItem.first << "_" << configItem.second;
        }
        result << "_inputShape=" << CommonTestUtils::vec2str(inputShape);
        result << "_weightstMinMax=(" << weightsMinMax.first << ".." << weightsMinMax.second << ")";
        result << "_levels=" << levels;

        return result.str();
    }

protected:
    void SetUp() override {
        InferenceEngine::Precision netPrecision;

        std::vector<size_t> inputShape;
        std::pair<float, float> weightsMinMax;
        size_t levels;
        std::tie(netPrecision, targetDevice, configuration, inputShape, weightsMinMax, levels) = this->GetParam();
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

        auto params = ngraph::builder::makeParams(ngPrc, {inputShape, inputShape});

        const size_t outChannels = 8;
        const size_t kernelSize = 8;
        auto weights = ngraph::builder::makeConstant<float>(ngPrc, {outChannels, inputShape[1], 1, kernelSize},
            CommonTestUtils::generate_float_numbers(outChannels * inputShape[1] * kernelSize,
                                                    weightsMinMax.first, weightsMinMax.second));
        auto weightsLowNode = ngraph::builder::makeConstant<float>(ngPrc, {1}, { weightsMinMax.first });
        auto weightsHighNode = ngraph::builder::makeConstant<float>(ngPrc, {1}, { weightsMinMax.second });
        auto weightsFQ = std::make_shared<ngraph::opset7::FakeQuantize>(weights,
            weightsLowNode, weightsHighNode, weightsLowNode, weightsHighNode, levels);

        auto conv1 = std::make_shared<ngraph::opset7::Convolution>(params[0], weightsFQ, std::vector<size_t>{ 1, 1 },
                                                                   std::vector<ptrdiff_t>{ 0, 0 }, std::vector<ptrdiff_t>{ 0, 0 },
                                                                   std::vector<size_t>{ 1, 1 }, ngraph::op::PadType::VALID);
        auto add1 = std::make_shared<ngraph::opset7::Add>(conv1,
            ngraph::builder::makeConstant<float>(ngPrc, {}, std::vector<float>{0.0f}));
        auto conv2 = std::make_shared<ngraph::opset7::Convolution>(params[1], weightsFQ, std::vector<size_t>{ 1, 1 },
                                                                   std::vector<ptrdiff_t>{ 0, 0 }, std::vector<ptrdiff_t>{ 0, 0 },
                                                                   std::vector<size_t>{ 1, 1 }, ngraph::op::PadType::VALID);
        auto add2 = std::make_shared<ngraph::opset7::Add>(conv2,
            ngraph::builder::makeConstant<float>(ngPrc, {}, std::vector<float>{0.0f}));

        auto outLowNode = ngraph::builder::makeConstant<float>(ngPrc, {1}, { -weightsMinMax.second * kernelSize * 10.0f });
        auto outHighNode = ngraph::builder::makeConstant<float>(ngPrc, {1}, { weightsMinMax.second * kernelSize * 10.0f });
        auto fq1 = std::make_shared<ngraph::opset7::FakeQuantize>(add1,
            outLowNode, outHighNode, outLowNode, outHighNode, levels);
        auto fq2 = std::make_shared<ngraph::opset7::FakeQuantize>(add2,
            outLowNode, outHighNode, outLowNode, outHighNode, levels);

        auto add3 = std::make_shared<ngraph::opset7::Add>(fq1, fq2);

        ngraph::ResultVector results{ std::make_shared<ngraph::opset7::Result>(add3)};
        function = std::make_shared<ngraph::Function>(results, params, "FQFusionWithMultipleWeights");
    }
};

TEST_P(FQFusionWithMultipleWeights, CompareWithRefImpl) {
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
    {1, 8, 1, 128}
};

const std::vector<std::pair<float, float>> weightsMinMax = {
    {-0.2, 0.2}
};

const std::vector<size_t> levels = {
    65535,
};

INSTANTIATE_TEST_SUITE_P(smoke_fq_fusion, FQFusionWithMultipleWeights,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(CommonTestUtils::DEVICE_GNA),
        ::testing::ValuesIn(configs),
        ::testing::ValuesIn(inputShape),
        ::testing::ValuesIn(weightsMinMax),
        ::testing::ValuesIn(levels)),
    FQFusionWithMultipleWeights::getTestCaseName);
} // namespace LayerTestsDefinitions