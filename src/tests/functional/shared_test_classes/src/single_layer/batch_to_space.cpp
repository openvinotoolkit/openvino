// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/builders.hpp"
#include "functional_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/single_layer/batch_to_space.hpp"

#include "functional_test_utils/plugin_cache.hpp"

namespace ov {
namespace test {
namespace subgraph {

std::string BatchToSpaceLayerTest::getTestCaseName(const testing::TestParamInfo<BatchToSpaceTestParams>& obj) {
    std::vector<InputShape> shapes;
    std::vector<int64_t> blockShape, cropsBegin, cropsEnd;
    ElementType netType, inType, outType;
    std::string targetName;
    std::tie(shapes, blockShape, cropsBegin, cropsEnd, netType, inType, outType, targetName) = obj.param;
    std::ostringstream result;
    result << "IS=(";
    for (const auto& shape : shapes) {
        result << CommonTestUtils::partialShape2str({shape.first}) << "_";
    }
    result << ")_TS=(";
    for (const auto& shape : shapes) {
        for (const auto& item : shape.second) {
            result << CommonTestUtils::vec2str(item) << "_";
        }
    }
    result << "netPRC=" << netType << "_";
    result << "inPRC=" << inType << "_";
    result << "outPRC=" << outType << "_";
    result << "BS=" << CommonTestUtils::vec2str(blockShape) << "_";
    result << "CB=" << CommonTestUtils::vec2str(cropsBegin) << "_";
    result << "CE=" << CommonTestUtils::vec2str(cropsEnd) << "_";
    result << "trgDev=" << targetName;
    return result.str();
}

void BatchToSpaceLayerTest::SetUp() {
    // w/a for myriad (cann't store 2 caches simultaneously)
    PluginCache::get().reset();

    std::vector<InputShape> shapes;
    std::vector<int64_t> blockShape, cropsBegin, cropsEnd;
    ElementType netType;
    std::string targetName;
    std::tie(shapes, blockShape, cropsBegin, cropsEnd, netType, inType, outType, targetDevice) = this->GetParam();

    init_input_shapes(shapes);

    auto params = ngraph::builder::makeDynamicParams(netType, {inputDynamicShapes.front()});
    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
    auto b2s = ngraph::builder::makeBatchToSpace(paramOuts[0], netType, blockShape, cropsBegin, cropsEnd);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(b2s)};
    function = std::make_shared<ngraph::Function>(results, params, "BatchToSpace");
}
} // namespace subgraph
} // namespace test
} // namespace ov
