// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/embedding_bag_offsets_sum.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "ngraph_functions/builders.hpp"

using namespace ov::test;

namespace LayerTestsDefinitions {

std::string EmbeddingBagOffsetsSumLayerTest::getTestCaseName(const testing::TestParamInfo<embeddingBagOffsetsSumLayerTestParamsSet>& obj) {
    embeddingBagOffsetsSumParams params;
    ov::test::ElementType netPrecision, indPrecision;
    std::string targetDevice;
    std::tie(params, netPrecision, indPrecision, targetDevice) = obj.param;

    ov::test::InputShape inputShapes;
    std::vector<size_t> indices, offsets;
    size_t defaultIndex;
    bool withWeights, withDefIndex;
    std::tie(inputShapes, indices, offsets, defaultIndex, withWeights, withDefIndex) = params;

    std::ostringstream result;
    result << "IS=" << inputShapes << "_";
    result << "I" << CommonTestUtils::vec2str(indices) << "_";
    result << "O" << CommonTestUtils::vec2str(offsets) << "_";
    result << "DI" << defaultIndex << "_";
    result << "WW" << withWeights << "_";
    result << "WDI" << withDefIndex << "_";
    result << "netPRC=" << netPrecision << "_";
    result << "indPRC=" << indPrecision << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void EmbeddingBagOffsetsSumLayerTest::SetUp() {
    embeddingBagOffsetsSumParams embParams;
    ov::test::ElementType netPrecision, indPrecision;
    std::tie(embParams, netPrecision, indPrecision, targetDevice) = this->GetParam();

    ov::test::InputShape inputShapes;
    std::vector<size_t> indices, offsets;
    bool withWeights, withDefIndex;
    size_t defaultIndex;
    std::tie(inputShapes, indices, offsets, defaultIndex, withWeights, withDefIndex) = embParams;

    init_input_shapes({ inputShapes });

    auto emb_table_node = std::make_shared<ngraph::opset1::Parameter>(netPrecision, inputShapes.first);
    ngraph::ParameterVector params = {emb_table_node};

    auto embBag = std::dynamic_pointer_cast<ngraph::opset3::EmbeddingBagOffsetsSum>(
            ngraph::builder::makeEmbeddingBagOffsetsSum(
                netPrecision, indPrecision, emb_table_node, indices, offsets, defaultIndex, withWeights, withDefIndex));
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(embBag)};
    function = std::make_shared<ngraph::Function>(results, params, "embeddingBagOffsetsSum");
}
}  // namespace LayerTestsDefinitions
