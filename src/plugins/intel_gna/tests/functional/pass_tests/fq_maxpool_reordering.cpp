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
                   std::pair<float, float>,             // Input Min and Max (before conv)
                   std::pair<float, float>,             // Input Min and Max (after conv)
                   size_t,                              // Levels
                   bool                                 // Reshape between FQ and Pooling
                   >
    fqMaxpoolReorderingParams;

namespace LayerTestsDefinitions {

class FQMaxpoolReordering : public testing::WithParamInterface<fqMaxpoolReorderingParams>,
                            public LayerTestsUtils::LayerTestsCommon {
    float inputDataMin1 = 0.0f;
    float inputDataMax1 = 0.0f;
    float inputDataMin2 = 0.0f;
    float inputDataMax2 = 0.0f;
    float inputDataResolution = 1.0f;

public:
    static std::string getTestCaseName(testing::TestParamInfo<fqMaxpoolReorderingParams> obj) {
        InferenceEngine::Precision netPrecision;
        std::string targetDevice;
        std::map<std::string, std::string> configuration;
        std::vector<size_t> inputShape;
        std::pair<float, float> inputMinMax1;
        std::pair<float, float> inputMinMax2;
        size_t levels = 0;
        bool reshape = false;
        std::tie(netPrecision, targetDevice, configuration, inputShape, inputMinMax1, inputMinMax2, levels, reshape) =
            obj.param;

        std::ostringstream result;
        result << "netPRC=" << netPrecision.name() << "_";
        result << "targetDevice=" << targetDevice << "_";
        for (auto const& configItem : configuration) {
            result << "_configItem=" << configItem.first << "_" << configItem.second;
        }
        result << "_inputShape=" << ov::test::utils::vec2str(inputShape);
        result << "_inputMinMax1=(" << inputMinMax1.first << ".." << inputMinMax1.second << ")";
        result << "_inputMinMax2=(" << inputMinMax2.first << ".." << inputMinMax2.second << ")";
        result << "_levels=" << levels;
        result << "_reshape=" << reshape;

        return result.str();
    }

    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo& info) const override {
        return FuncTestUtils::createAndFillBlob(info.getTensorDesc(),
                                                inputDataMax1 - inputDataMin1,
                                                inputDataMin1,
                                                1 / inputDataResolution);
    }

protected:
    void SetUp() override {
        InferenceEngine::Precision netPrecision;

        std::vector<size_t> inputShape;
        std::pair<float, float> inputMinMax1;
        std::pair<float, float> inputMinMax2;
        size_t levels = 0;
        bool reshape = false;
        std::tie(netPrecision, targetDevice, configuration, inputShape, inputMinMax1, inputMinMax2, levels, reshape) =
            this->GetParam();
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

        std::tie(inputDataMin1, inputDataMax1) = inputMinMax1;
        std::tie(inputDataMin2, inputDataMax2) = inputMinMax2;
        auto inputLowNode1 = ngraph::builder::makeConstant<float>(ngPrc, {1}, {inputDataMin1});
        auto inputHighNode1 = ngraph::builder::makeConstant<float>(ngPrc, {1}, {inputDataMax1});
        auto inputLowNode2 = ngraph::builder::makeConstant<float>(ngPrc, {1}, {inputDataMin2});
        auto inputHighNode2 = ngraph::builder::makeConstant<float>(ngPrc, {1}, {inputDataMax2});

        ov::ParameterVector inputVector{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};

        auto inputFQ = std::make_shared<ngraph::opset1::FakeQuantize>(inputVector[0],
                                                                      inputLowNode1,
                                                                      inputHighNode1,
                                                                      inputLowNode1,
                                                                      inputHighNode1,
                                                                      levels);

        auto filterWeightsNode = ngraph::builder::makeConstant<float>(ngPrc, {8, inputShape[1], 1, 8}, {1.0f});
        auto convLowNode = ngraph::builder::makeConstant(ngraph::element::f32,
                                                         std::vector<size_t>{1},
                                                         std::vector<float>{inputDataMin1 * 35});
        auto convHighNode = ngraph::builder::makeConstant(ngraph::element::f32,
                                                          std::vector<size_t>{1},
                                                          std::vector<float>{inputDataMax1 * 35});
        auto convWeightsFQNode = std::make_shared<ngraph::opset1::FakeQuantize>(filterWeightsNode,
                                                                                convLowNode,
                                                                                convHighNode,
                                                                                convLowNode,
                                                                                convHighNode,
                                                                                levels);
        auto convWeightsFQ = std::dynamic_pointer_cast<ngraph::opset1::FakeQuantize>(convWeightsFQNode);

        auto conv = std::make_shared<ngraph::opset1::Convolution>(inputFQ,
                                                                  convWeightsFQ,
                                                                  std::vector<size_t>{1, 1},
                                                                  std::vector<ptrdiff_t>{0, 0},
                                                                  std::vector<ptrdiff_t>{0, 0},
                                                                  std::vector<size_t>{1, 1},
                                                                  ngraph::op::PadType::VALID);
        auto biasesWeightsNode = ngraph::builder::makeConstant(ngPrc, {}, std::vector<float>{0.0f});
        auto add = std::make_shared<ngraph::opset1::Add>(conv, biasesWeightsNode);

        auto convFQNode = std::make_shared<ngraph::opset1::FakeQuantize>(add,
                                                                         inputLowNode2,
                                                                         inputHighNode2,
                                                                         inputLowNode2,
                                                                         inputHighNode2,
                                                                         levels);

        std::shared_ptr<ngraph::Node> node_before_pooling = convFQNode;
        if (reshape) {
            const auto& shape = conv->get_output_shape(0);
            size_t total = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
            auto reshapeConst1 =
                ngraph::builder::makeConstant(ngraph::element::i64, std::vector<size_t>{2}, ngraph::Shape{1, total});
            auto reshapeNode1 = std::make_shared<ngraph::opset1::Reshape>(convFQNode, reshapeConst1, false);
            auto reshapeConst2 = ngraph::builder::makeConstant(ngraph::element::i64, std::vector<size_t>{4}, shape);
            auto reshapeNode2 = std::make_shared<ngraph::opset1::Reshape>(reshapeNode1, reshapeConst2, false);
            node_before_pooling = reshapeNode2;
        }

        OPENVINO_SUPPRESS_DEPRECATED_START
        auto maxpool = ngraph::builder::makePooling(node_before_pooling,
                                                    {1, 2},
                                                    {0, 0},
                                                    {0, 0},
                                                    {1, 2},
                                                    ngraph::op::RoundingType::FLOOR,
                                                    ngraph::op::PadType::VALID,
                                                    false,
                                                    ngraph::helpers::PoolingTypes::MAX);
        OPENVINO_SUPPRESS_DEPRECATED_END

        ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(maxpool)};
        function = std::make_shared<ngraph::Function>(results, inputVector, "FQMaxPoolReorder");
    }
};

TEST_P(FQMaxpoolReordering, CompareWithRefImpl) {
    Run();
};

const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                               InferenceEngine::Precision::FP16};

const std::vector<std::map<std::string, std::string>> configs = {{
    {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
}};

const std::vector<std::vector<size_t>> inputShape = {{1, 1, 1, 1024}, {1, 8, 1, 168}};

const std::vector<std::pair<float, float>> inputMinMax = {{-0.5, 0.5}, {-2, 2}, {-8, 8}, {-5, 5}, {-17.5, 17.5}};

const std::vector<size_t> levels = {65535};

INSTANTIATE_TEST_SUITE_P(smoke_fq_maxpool_reordering,
                         FQMaxpoolReordering,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs),
                                            ::testing::ValuesIn(inputShape),
                                            ::testing::ValuesIn(inputMinMax),
                                            ::testing::ValuesIn(inputMinMax),
                                            ::testing::ValuesIn(levels),
                                            ::testing::ValuesIn(std::vector<bool>{true, false})),
                         FQMaxpoolReordering::getTestCaseName);
}  // namespace LayerTestsDefinitions
