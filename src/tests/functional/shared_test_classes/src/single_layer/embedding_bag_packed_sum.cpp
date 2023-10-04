// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/embedding_bag_packed_sum.hpp"
#include "ov_models/builders.hpp"

namespace LayerTestsDefinitions {

std::string EmbeddingBagPackedSumLayerTest::getTestCaseName(const testing::TestParamInfo<embeddingBagPackedSumLayerTestParamsSet>& obj) {
    embeddingBagPackedSumParams params;
    InferenceEngine::Precision netPrecision, indPrecision;
    std::string targetDevice;
    std::tie(params, netPrecision, indPrecision, targetDevice) = obj.param;
    std::vector<size_t> embTableShape;
    std::vector<std::vector<size_t>> indices;
    bool withWeights;
    std::tie(embTableShape, indices, withWeights) = params;

    std::ostringstream result;
    result << "ETS=" << ov::test::utils::vec2str(embTableShape) << "_";
    result << "I" << ov::test::utils::vec2str(indices) << "_";
    result << "WW" << withWeights << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "indPRC=" << indPrecision.name() << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void EmbeddingBagPackedSumLayerTest::SetUp() {
    embeddingBagPackedSumParams embParams;
    auto netPrecision = InferenceEngine::Precision::UNSPECIFIED;
    auto indPrecision = netPrecision;
    std::tie(embParams, netPrecision, indPrecision, targetDevice) = this->GetParam();
    std::vector<size_t> embTableShape;
    std::vector<std::vector<size_t>> indices;
    bool withWeights;
    std::tie(embTableShape, indices, withWeights) = embParams;
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto ngIdxPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(indPrecision);

    auto emb_table_node = std::make_shared<ngraph::opset1::Parameter>(ngPrc, ngraph::Shape(embTableShape));
    ngraph::ParameterVector params = {emb_table_node};

    auto embBag = std::dynamic_pointer_cast<ngraph::opset3::EmbeddingBagPackedSum>(
            ngraph::builder::makeEmbeddingBagPackedSum(
                ngPrc, ngIdxPrc, emb_table_node, indices, withWeights));
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(embBag)};
    function = std::make_shared<ngraph::Function>(results, params, "embeddingBagPackedSum");
}
}  // namespace LayerTestsDefinitions
