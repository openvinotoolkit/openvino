// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/embedding_segments_sum.hpp"
#include "ov_models/builders.hpp"


namespace LayerTestsDefinitions {

std::string EmbeddingSegmentsSumLayerTest::getTestCaseName(const testing::TestParamInfo<embeddingSegmentsSumLayerTestParamsSet>& obj) {
    embeddingSegmentsSumParams params;
    InferenceEngine::Precision netPrecision, indPrecision;
    std::string targetDevice;
    std::tie(params, netPrecision, indPrecision, targetDevice) = obj.param;
    std::vector<size_t> embTableShape, indices, segmentIds;
    size_t numSegments, defaultIndex;
    bool withWeights, withDefIndex;
    std::tie(embTableShape, indices, segmentIds, numSegments, defaultIndex, withWeights, withDefIndex) = params;

    std::ostringstream result;
    result << "ETS=" << ov::test::utils::vec2str(embTableShape) << "_";
    result << "I"  << ov::test::utils::vec2str(indices) << "_";
    result << "SI" << ov::test::utils::vec2str(segmentIds) << "_";
    result << "NS" << numSegments << "_";
    result << "DI" << defaultIndex << "_";
    result << "WW" << withWeights << "_";
    result << "WDI" << withDefIndex << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "indPRC=" << indPrecision.name() << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void EmbeddingSegmentsSumLayerTest::SetUp() {
    embeddingSegmentsSumParams embParams;
    auto netPrecision = InferenceEngine::Precision::UNSPECIFIED;
    auto indPrecision = netPrecision;
    std::tie(embParams, netPrecision, indPrecision, targetDevice) = this->GetParam();
    std::vector<size_t> embTableShape, indices, segmentIds;
    bool withWeights, withDefIndex;
    size_t numSegments, defaultIndex;
    std::tie(embTableShape, indices, segmentIds, numSegments, defaultIndex, withWeights, withDefIndex) = embParams;
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto ngIdxPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(indPrecision);

    auto emb_table_node = std::make_shared<ngraph::opset1::Parameter>(ngPrc, ngraph::Shape(embTableShape));
    ngraph::ParameterVector params = {emb_table_node};

    auto embBag = std::dynamic_pointer_cast<ngraph::opset3::EmbeddingSegmentsSum>(
            ngraph::builder::makeEmbeddingSegmentsSum(
                ngPrc, ngIdxPrc, emb_table_node, indices, segmentIds, numSegments, defaultIndex, withWeights, withDefIndex));
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(embBag)};
    function = std::make_shared<ngraph::Function>(results, params, "embeddingSegmentsSum");
}
}  // namespace LayerTestsDefinitions
