// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <vector>
#include <string>
#include <memory>

#include "single_layer_tests/embedding_segments_sum.hpp"

#include "functional_test_utils/layer_test_utils.hpp"
#include "ngraph_functions/builders.hpp"


namespace LayerTestsDefinitions {

std::string EmbeddingSegmentsSumLayerTest::getTestCaseName(testing::TestParamInfo<embeddingSegmentsSumLayerTestParamsSet> obj) {
    embeddingSegmentsSumParams params;
    InferenceEngine::Precision netPrecision;
    std::string targetDevice;
    std::tie(params, netPrecision, targetDevice) = obj.param;
    std::vector<size_t> emb_table_shape, indices, segment_ids;
    size_t num_segments, default_index;
    bool with_weights, with_def_index;
    std::tie(emb_table_shape, indices, segment_ids, num_segments, default_index, with_weights, with_def_index) = params;

    std::ostringstream result;
    result << "ETS=" << CommonTestUtils::vec2str(emb_table_shape) << "_";
    result << "I"  << CommonTestUtils::vec2str(indices) << "_";
    result << "SI" << CommonTestUtils::vec2str(segment_ids) << "_";
    result << "NS" << num_segments << "_";
    result << "DI" << default_index << "_";
    result << "WW" << with_weights << "_";
    result << "WDI" << with_def_index << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void EmbeddingSegmentsSumLayerTest::SetUp() {
    embeddingSegmentsSumParams embParams;
    auto netPrecision = InferenceEngine::Precision::UNSPECIFIED;
    std::tie(embParams, netPrecision, targetDevice) = this->GetParam();
    std::vector<size_t> emb_table_shape, indices, segment_ids;
    bool with_weights, with_def_index;
    size_t num_segments, default_index;
    std::tie(emb_table_shape, indices, segment_ids, num_segments, default_index, with_weights, with_def_index) = embParams;
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto ngIdxPrc = ngraph::element::Type_t::i64;

    auto emb_table_node = std::make_shared<ngraph::opset1::Parameter>(ngPrc, ngraph::Shape(emb_table_shape));
    ngraph::ParameterVector params = {emb_table_node};

    auto embBag = std::dynamic_pointer_cast<ngraph::opset3::EmbeddingSegmentsSum>(
            ngraph::builder::makeEmbeddingSegmentsSum(
                ngPrc, ngIdxPrc, emb_table_node, indices, segment_ids, num_segments, default_index, with_weights, with_def_index));
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(embBag)};
    function = std::make_shared<ngraph::Function>(results, params, "embeddingSegmentsSum");
}

TEST_P(EmbeddingSegmentsSumLayerTest, CompareWithRefs) {
    Run();
}

}  // namespace LayerTestsDefinitions
