// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/embedding_bag_offsets_sum.hpp"
#include "ngraph_functions/builders.hpp"

namespace LayerTestsDefinitions {

std::string EmbeddingBagOffsetsSumLayerTest::getTestCaseName(const testing::TestParamInfo<embeddingBagOffsetsSumLayerTestParamsSet>& obj) {
    embeddingBagOffsetsSumParams params;
    InferenceEngine::Precision netPrecision, indPrecision;
    std::string targetDevice;
    std::tie(params, netPrecision, indPrecision, targetDevice) = obj.param;
    std::vector<size_t> embTableShape, indices, offsets;
    size_t defaultIndex;
    bool withWeights, withDefIndex;
    std::tie(embTableShape, indices, offsets, defaultIndex, withWeights, withDefIndex) = params;

    std::ostringstream result;
    result << "ETS=" << CommonTestUtils::vec2str(embTableShape) << "_";
    result << "I" << CommonTestUtils::vec2str(indices) << "_";
    result << "O" << CommonTestUtils::vec2str(offsets) << "_";
    result << "DI" << defaultIndex << "_";
    result << "WW" << withWeights << "_";
    result << "WDI" << withDefIndex << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "indPRC=" << indPrecision.name() << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void EmbeddingBagOffsetsSumLayerTest::SetUp() {
    embeddingBagOffsetsSumParams embParams;
    auto netPrecision = InferenceEngine::Precision::UNSPECIFIED;
    auto indPrecision = netPrecision;
    std::tie(embParams, netPrecision, indPrecision, targetDevice) = this->GetParam();
    std::vector<size_t> embTableShape, indices, offsets;
    bool withWeights, withDefIndex;
    size_t defaultIndex;
    std::tie(embTableShape, indices, offsets, defaultIndex, withWeights, withDefIndex) = embParams;
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto ngIdxPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(indPrecision);

    auto emb_table_node = std::make_shared<ngraph::opset1::Parameter>(ngPrc, ngraph::Shape(embTableShape));
    ngraph::ParameterVector params = {emb_table_node};

    auto embBag = std::dynamic_pointer_cast<ngraph::opset3::EmbeddingBagOffsetsSum>(
            ngraph::builder::makeEmbeddingBagOffsetsSum(
                ngPrc, ngIdxPrc, emb_table_node, indices, offsets, defaultIndex, withWeights, withDefIndex));
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(embBag)};
    function = std::make_shared<ngraph::Function>(results, params, "embeddingBagOffsetsSum");
}
}  // namespace LayerTestsDefinitions
