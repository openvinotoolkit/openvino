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
    std::vector<size_t>                 // Input Shape
> convertMatmulToPointwiseConvParams;

typedef std::tuple<
    InferenceEngine::Precision,         // Network Precision
    std::string,                        // Target Device
    std::map<std::string, std::string>, // Configuration
    std::vector<size_t>,                // Input Shape
    std::pair<float, float>             // Input Min and Max
> convertMatmulToPointwiseConvWithFqParams;

namespace LayerTestsDefinitions {

class ConvertMatmulToPointwiseConv : public testing::WithParamInterface<convertMatmulToPointwiseConvParams>,
    public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<convertMatmulToPointwiseConvParams> obj) {
        InferenceEngine::Precision netPrecision;
        std::string targetDevice;
        std::map<std::string, std::string> configuration;
        std::vector<size_t> inputShape;
        std::tie(netPrecision, targetDevice, configuration, inputShape) = obj.param;

        std::ostringstream result;
        result << "netPRC=" << netPrecision.name() << "_";
        result << "targetDevice=" << targetDevice << "_";
        for (auto const& configItem : configuration) {
            result << "_configItem=" << configItem.first << "_" << configItem.second;
        }
        result << "_inputShape=" << CommonTestUtils::vec2str(inputShape);
        return result.str();
    }

    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo& info) const {
        InferenceEngine::Blob::Ptr blob = make_blob_with_precision(info.getTensorDesc());
        blob->allocate();

        auto* rawBlobDataPtr = blob->buffer().as<float*>();
        std::vector<float> values = CommonTestUtils::generate_float_numbers(blob->size(), -0.2f, 0.2f);
        for (size_t i = 0; i < blob->size(); i++) {
            rawBlobDataPtr[i] = values[i];
        }
        return blob;
    }

protected:
    void SetUp() override {
        InferenceEngine::Precision netPrecision;
        std::vector<size_t> inputShape;
        std::tie(netPrecision, targetDevice, configuration, inputShape) = this->GetParam();
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

        auto params = ngraph::builder::makeParams(ngPrc, {inputShape});

        size_t batch = inputShape[inputShape.size() - 2];
        size_t elemNum = inputShape[inputShape.size() - 1];
        std::vector<float> weights = CommonTestUtils::generate_float_numbers(elemNum * elemNum, -0.1f, 0.1f);
        auto weightsNode = std::make_shared<ngraph::opset7::Constant>(ngPrc, ngraph::Shape{elemNum, elemNum}, weights);
        auto matmul = ngraph::builder::makeMatMul(params[0], weightsNode, false, true);

        auto bias = ngraph::builder::makeConstant(ngPrc, std::vector<size_t>{1, batch, 1}, std::vector<float>{1.0f});
        auto add = ngraph::builder::makeEltwise(matmul, bias, ngraph::helpers::EltwiseTypes::ADD);

        auto pattern = std::make_shared<ngraph::opset7::Constant>(ngraph::element::Type_t::i64,
            ngraph::Shape{ inputShape.size() }, inputShape);
        auto reshape = std::make_shared<ngraph::opset7::Reshape>(matmul, pattern, false);
        auto relu = std::make_shared<ngraph::opset7::Relu>(reshape);

        ngraph::ResultVector results{ std::make_shared<ngraph::opset7::Result>(relu)};
        function = std::make_shared<ngraph::Function>(results, params, "ConvertMatmulToPointwiseConv");
    }
};

class ConvertMatmulToPointwiseConvWithFq : public testing::WithParamInterface<convertMatmulToPointwiseConvWithFqParams>,
    public LayerTestsUtils::LayerTestsCommon {
    float inputDataMin = -10.0f;
    float inputDataMax = 10.0f;
    float inputDataResolution = 1.0f;

public:
    static std::string getTestCaseName(testing::TestParamInfo<convertMatmulToPointwiseConvWithFqParams> obj) {
        InferenceEngine::Precision netPrecision;
        std::string targetDevice;
        std::map<std::string, std::string> configuration;
        std::vector<size_t> inputShape;
        std::pair<float, float> inputMinMax;
        std::tie(netPrecision, targetDevice, configuration, inputShape, inputMinMax) = obj.param;

        std::ostringstream result;
        result << "netPRC=" << netPrecision.name() << "_";
        result << "targetDevice=" << targetDevice << "_";
        for (auto const& configItem : configuration) {
            result << "_configItem=" << configItem.first << "_" << configItem.second;
        }
        result << "_inputShape=" << CommonTestUtils::vec2str(inputShape);
        result << "_inputMinMax=(" << inputMinMax.first << ".." << inputMinMax.second << ")";
        return result.str();
    }

    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo& info) const {
        return FuncTestUtils::createAndFillBlob(info.getTensorDesc(), inputDataMax - inputDataMin, inputDataMin,
            1 / inputDataResolution);
    }

protected:
    void SetUp() override {
        InferenceEngine::Precision netPrecision;
        std::vector<size_t> inputShape;
        std::pair<float, float> inputMinMax;
        std::tie(netPrecision, targetDevice, configuration, inputShape, inputMinMax) = this->GetParam();
        std::tie(inputDataMin, inputDataMax) = inputMinMax;
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

        auto params = ngraph::builder::makeParams(ngPrc, {inputShape});

        auto inputLowNode = ngraph::builder::makeConstant(ngPrc, std::vector<size_t>{ 1 },
            std::vector<float>{ inputDataMin });
        auto inputHighNode = ngraph::builder::makeConstant(ngPrc, std::vector<size_t>{ 1 },
            std::vector<float>{ inputDataMax });
        auto inputFQ = std::make_shared<ngraph::opset7::FakeQuantize>(params[0],
            inputLowNode, inputHighNode, inputLowNode, inputHighNode, UINT16_MAX);

        size_t elemNum = inputShape[inputShape.size() - 1];

        const float weightsMin = -0.2f;
        const float weightsMax = 0.2f;
        std::vector<float> weights = CommonTestUtils::generate_float_numbers(elemNum * elemNum, weightsMin, weightsMax);
        auto weightsNode = std::make_shared<ngraph::opset7::Constant>(ngPrc, ngraph::Shape{elemNum, elemNum}, weights);
        auto weightsLowNode = ngraph::builder::makeConstant(ngPrc, std::vector<size_t>{ 1 },
            std::vector<float>{ weightsMin });
        auto weightsHighNode = ngraph::builder::makeConstant(ngPrc, std::vector<size_t>{ 1 },
              std::vector<float>{ weightsMax });
        auto weightsFQNode = std::make_shared<ngraph::opset7::FakeQuantize>(weightsNode,
            weightsLowNode, weightsHighNode, weightsLowNode, weightsHighNode, UINT16_MAX);
        auto matmul = ngraph::builder::makeMatMul(inputFQ, weightsFQNode, false, true);

        auto bias = ngraph::builder::makeConstant(ngPrc, std::vector<size_t>{1, 1, 1}, std::vector<float>{1.0f});
        auto add = ngraph::builder::makeEltwise(matmul, bias, ngraph::helpers::EltwiseTypes::ADD);

        auto outputLowNode = ngraph::builder::makeConstant(ngPrc, std::vector<size_t>{ 1 },
            std::vector<float>{ -inputDataMax * weightsMax *  elemNum });
        auto outputHighNode = ngraph::builder::makeConstant(ngPrc, std::vector<size_t>{ 1 },
            std::vector<float>{ inputDataMax * weightsMax * elemNum });
        auto outputFQ = std::make_shared<ngraph::opset7::FakeQuantize>(add,
            outputLowNode, outputHighNode, outputLowNode, outputHighNode, UINT16_MAX);

        auto pattern = std::make_shared<ngraph::opset7::Constant>(ngraph::element::Type_t::i64,
            ngraph::Shape{ inputShape.size() }, inputShape);
        auto reshape = std::make_shared<ngraph::opset7::Reshape>(outputFQ, pattern, false);

        auto relu = std::make_shared<ngraph::opset7::Relu>(reshape);

        ngraph::ResultVector results{ std::make_shared<ngraph::opset7::Result>(relu)};
        function = std::make_shared<ngraph::Function>(results, params, "ConvertMatmulToPointwiseConv");
    }
};

TEST_P(ConvertMatmulToPointwiseConv, CompareWithRefImpl) {
    Run();
};

TEST_P(ConvertMatmulToPointwiseConvWithFq, CompareWithRefImpl) {
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
    {1, 64, 64},
    {1, 256, 128},
    {1, 512, 128}
};

const std::vector<std::pair<float, float>> fqStats = {
    {-0.5, 0.5}
};

INSTANTIATE_TEST_CASE_P(smoke_ConvertMatmulToPointwiseConvTest, ConvertMatmulToPointwiseConv,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(CommonTestUtils::DEVICE_GNA),
        ::testing::ValuesIn(configs),
        ::testing::ValuesIn(inputShape)),
    ConvertMatmulToPointwiseConv::getTestCaseName);

// Issue 55662
INSTANTIATE_TEST_CASE_P(DISABLED_smoke_ConvertMatmulToPointwiseConvTest, ConvertMatmulToPointwiseConvWithFq,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(CommonTestUtils::DEVICE_GNA),
        ::testing::ValuesIn(configs),
        ::testing::ValuesIn(inputShape),
        ::testing::ValuesIn(fqStats)),
    ConvertMatmulToPointwiseConvWithFq::getTestCaseName);

} // namespace LayerTestsDefinitions
