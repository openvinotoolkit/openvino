// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <vector>
#include <string>
#include <memory>

#include "single_layer_tests/embedding_bag_packed_sum.hpp"

#include "functional_test_utils/layer_test_utils.hpp"
#include "ngraph_functions/builders.hpp"


namespace LayerTestsDefinitions {

std::string EmbeddingBagPackedSumLayerTest::getTestCaseName(testing::TestParamInfo<embeddingBagPackedSumLayerTestParamsSet> obj) {
    embeddingBagPackedSumParams params;
    InferenceEngine::Precision netPrecision;
    std::string targetDevice;
    std::tie(params, netPrecision, targetDevice) = obj.param;
    std::vector<size_t> emb_table_shape;
    std::vector<std::vector<size_t>> indices;
    bool with_weights;
    std::tie(emb_table_shape, indices, with_weights) = params;

    std::ostringstream result;
    result << "ETS=" << CommonTestUtils::vec2str(emb_table_shape) << "_";
    result << "I" << CommonTestUtils::vec2str(indices) << "_";
    result << "WW" << with_weights << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void EmbeddingBagPackedSumLayerTest::SetUp() {
    embeddingBagPackedSumParams embParams;
    auto netPrecision = InferenceEngine::Precision::UNSPECIFIED;
    std::tie(embParams, netPrecision, targetDevice) = this->GetParam();
    std::vector<size_t> emb_table_shape;
    std::vector<std::vector<size_t>> indices;
    bool with_weights;
    std::tie(emb_table_shape, indices, with_weights) = embParams;
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto ngIdxPrc = ngraph::element::Type_t::i64;

    auto emb_table_node = std::make_shared<ngraph::opset1::Parameter>(ngPrc, ngraph::Shape(emb_table_shape));
    ngraph::ParameterVector params = {emb_table_node};

    auto embBag = std::dynamic_pointer_cast<ngraph::opset3::EmbeddingBagPackedSum>(
            ngraph::builder::makeEmbeddingBagPackedSum(
                ngPrc, ngIdxPrc, emb_table_node, indices, with_weights));
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(embBag)};
    function = std::make_shared<ngraph::Function>(results, params, "embeddingBagPackedSum");
}

TEST_P(EmbeddingBagPackedSumLayerTest, CompareWithRefs) {
    Run();
}
}  // namespace LayerTestsDefinitions
