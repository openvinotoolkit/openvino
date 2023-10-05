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
                   std::pair<float, float>,             // Weights values
                   std::vector<size_t>                  // Input shapes
                   >
    matmulParams;

namespace LayerTestsDefinitions {

class PerchannelQuantTest : public testing::WithParamInterface<matmulParams>, public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<matmulParams> obj) {
        InferenceEngine::Precision netPrecision;
        std::string targetDevice;
        std::map<std::string, std::string> configuration;
        std::pair<float, float> weightsValues;
        std::vector<size_t> inputShape;
        std::tie(netPrecision, targetDevice, configuration, weightsValues, inputShape) = obj.param;

        std::ostringstream result;
        result << "netPRC=" << netPrecision.name() << "_";
        result << "targetDevice=" << targetDevice << "_";
        for (auto const& configItem : configuration) {
            result << "_configItem=" << configItem.first << "_" << configItem.second;
        }
        result << "_range=(" << weightsValues.first << ", " << weightsValues.second << ")";
        result << "_IS=(" << ov::test::utils::vec2str(inputShape) << ")";

        return result.str();
    }

protected:
    void SetUp() override {
        InferenceEngine::Precision netPrecision;
        std::pair<float, float> weightsValues;
        std::vector<size_t> inputShape;
        std::tie(netPrecision, targetDevice, configuration, weightsValues, inputShape) = this->GetParam();
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

        const ngraph::Shape constShape = {inputShape.back(), inputShape.back()};
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};

        std::vector<float> weights;
        std::vector<float> weightsMin, weightsMax;
        for (int i = 0; i < constShape.front(); ++i) {
            // multiplier to increase weights ranges difference between different channels
            float mul = (i % 2 ? 1.0 : 0.001);
            float rowMin = weightsValues.first * mul;
            float rowMax = weightsValues.second * mul;
            auto rowWeights = ov::test::utils::generate_float_numbers(constShape.back(), rowMin, rowMax);
            weights.insert(std::end(weights), std::begin(rowWeights), std::end(rowWeights));
            weightsMin.push_back(rowMin);
            weightsMax.push_back(rowMax);
        }

        auto constant = ngraph::builder::makeConstant<float>(ngPrc, constShape, weights);
        auto wLowNode = ngraph::builder::makeConstant<float>(ngPrc, {constShape.front()}, {weightsMin});
        auto wHighNode = ngraph::builder::makeConstant<float>(ngPrc, {constShape.front()}, {weightsMax});
        auto wFq = std::make_shared<ngraph::opset8::FakeQuantize>(constant,
                                                                  wLowNode,
                                                                  wHighNode,
                                                                  wLowNode,
                                                                  wHighNode,
                                                                  std::numeric_limits<uint8_t>::max() - 1);
        auto matmul = std::make_shared<ngraph::opset8::MatMul>(params[0], wFq, false, true);

        ngraph::ResultVector results{std::make_shared<ngraph::opset8::Result>(matmul)};
        function = std::make_shared<ngraph::Function>(results, params, "PerchannelQuantTest");
    }
};

TEST_P(PerchannelQuantTest, CompareWithRefImpl) {
    LoadNetwork();
    GenerateInputs();
    Infer();
    auto results = GetOutputs();
    size_t size = results.front()->size();
    auto memory = InferenceEngine::as<InferenceEngine::MemoryBlob>(results.front());
    IE_ASSERT(memory);
    const auto lockedMemory = memory->wmap();
    const auto actualBuffer = lockedMemory.as<const float*>();

    // check that outputs haven't been zero out by a channel multilplier
    for (size_t i = 0; i < size; ++i) {
        if (actualBuffer[i] == 0.0) {
            IE_THROW() << "Unexpected 0 output value";
        }
    }
};

const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                               InferenceEngine::Precision::FP16};

const std::vector<std::map<std::string, std::string>> configs = {{{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}}};

const std::vector<std::pair<float, float>> weightsValues = {{-0.1, 0.1}, {-1.0, 1.0}, {-10.0, 10.0}};

const std::vector<std::vector<size_t>> inputShapes = {{1, 128}, {1, 38}, {1, 8}};

INSTANTIATE_TEST_SUITE_P(smoke_base,
                         PerchannelQuantTest,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs),
                                            ::testing::ValuesIn(weightsValues),
                                            ::testing::ValuesIn(inputShapes)),
                         PerchannelQuantTest::getTestCaseName);
}  // namespace LayerTestsDefinitions
