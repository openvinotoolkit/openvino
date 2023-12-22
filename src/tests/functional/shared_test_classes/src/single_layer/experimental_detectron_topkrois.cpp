// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_models/builders.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>
#include "shared_test_classes/single_layer/experimental_detectron_topkrois.hpp"

namespace ov {
namespace test {
namespace subgraph {

std::string ExperimentalDetectronTopKROIsLayerTest::getTestCaseName(const testing::TestParamInfo<ExperimentalDetectronTopKROIsTestParams>& obj) {
    std::vector<InputShape> inputShapes;
    int64_t maxRois;
    ElementType netPrecision;
    std::string targetName;
    std::tie(inputShapes, maxRois, netPrecision, targetName) = obj.param;

    std::ostringstream result;
    if (inputShapes.front().first.size() != 0) {
        result << "IS=(";
        for (const auto &shape : inputShapes) {
            result << ov::test::utils::partialShape2str({shape.first}) << "_";
        }
        result.seekp(-1, result.cur);
        result << ")_";
    }
    result << "TS=";
    for (const auto& shape : inputShapes) {
        for (const auto& item : shape.second) {
            result << ov::test::utils::vec2str(item) << "_";
        }
    }
    result << "maxRois=" << maxRois << "_";
    result << "netPRC=" << netPrecision << "_";
    result << "trgDev=" << targetName;
    return result.str();
}

void ExperimentalDetectronTopKROIsLayerTest::SetUp() {
    std::vector<InputShape> inputShapes;
    int64_t maxRois;
    ElementType netPrecision;
    std::string targetName;
    std::tie(inputShapes, maxRois, netPrecision, targetName) = this->GetParam();

    inType = outType = netPrecision;
    targetDevice = targetName;

    init_input_shapes(inputShapes);

    ov::ParameterVector params;
    for (auto&& shape : inputDynamicShapes)
        params.push_back(std::make_shared<ov::op::v0::Parameter>(netPrecision, shape));

    auto experimentalDetectronTopKROIs = std::make_shared<ov::op::v6::ExperimentalDetectronTopKROIs>(params[0], params[1], maxRois);
    function = std::make_shared<ov::Model>(ov::OutputVector {experimentalDetectronTopKROIs->output(0)}, "ExperimentalDetectronTopKROIs");
}
} // namespace subgraph
} // namespace test
} // namespace ov
