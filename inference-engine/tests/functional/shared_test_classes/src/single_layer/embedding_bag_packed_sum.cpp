// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/embedding_bag_packed_sum.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "ngraph_functions/builders.hpp"

using namespace ov::test;

namespace LayerTestsDefinitions {

std::string EmbeddingBagPackedSumLayerTest::getTestCaseName(const testing::TestParamInfo<embeddingBagPackedSumLayerTestParamsSet>& obj) {
    embeddingBagPackedSumParams params;
    ov::test::ElementType netPrecision, indPrecision;
    std::string targetDevice;
    std::tie(params, netPrecision, indPrecision, targetDevice) = obj.param;

    ov::test::InputShape inputShapes;
    std::vector<std::vector<size_t>> indices;
    bool withWeights;
    std::tie(inputShapes, indices, withWeights) = params;

    std::ostringstream result;
    result << "IS=" << inputShapes << "_";
    result << "I" << CommonTestUtils::vec2str(indices) << "_";
    result << "WW" << withWeights << "_";
    result << "netPRC=" << netPrecision << "_";
    result << "indPRC=" << indPrecision << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void EmbeddingBagPackedSumLayerTest::SetUp() {
    embeddingBagPackedSumParams embParams;
    ov::test::ElementType netPrecision, indPrecision;
    std::tie(embParams, netPrecision, indPrecision, targetDevice) = this->GetParam();

    ov::test::InputShape inputShapes;
    std::vector<std::vector<size_t>> indices;
    bool withWeights;
    std::tie(inputShapes, indices, withWeights) = embParams;

    init_input_shapes({ inputShapes });

    auto emb_table_node = std::make_shared<ngraph::opset1::Parameter>(netPrecision, inputShapes.first);
    ngraph::ParameterVector params = {emb_table_node};

    auto embBag = std::dynamic_pointer_cast<ngraph::opset3::EmbeddingBagPackedSum>(
            ngraph::builder::makeEmbeddingBagPackedSum(
                netPrecision, indPrecision, emb_table_node, indices, withWeights));
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(embBag)};
    function = std::make_shared<ngraph::Function>(results, params, "embeddingBagPackedSum");
}
}  // namespace LayerTestsDefinitions
