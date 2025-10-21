// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/common_utils.hpp"
#include "snippets/two_inputs_and_outputs.hpp"
#include "subgraph_simple.hpp"

namespace ov {
namespace test {
namespace snippets {

std::string TwoInputsAndOutputs::getTestCaseName(const testing::TestParamInfo<ov::test::snippets::TwoInputsAndOutputsParams>& obj) {
    const auto& [inputShapes, num_nodes, num_subgraphs, targetDevice] = obj.param;

    std::ostringstream result;
    for (size_t i = 0; i < inputShapes.size(); ++i) {
        result << "IS[" << i << "]=" << ov::test::utils::partialShape2str({inputShapes[i].first}) << "_";
        result << "TS[" << i << "]=";
        for (const auto& shape : inputShapes[i].second) {
            result << "(" << ov::test::utils::vec2str(shape) << ")_";
        }
    }
    result << "#N=" << num_nodes << "_";
    result << "#S=" << num_subgraphs << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void TwoInputsAndOutputs::SetUp() {
    const auto& [inputShape, _ref_num_nodes, _ref_num_subgraphs, _targetDevice] = this->GetParam();
    ref_num_nodes = _ref_num_nodes;
    ref_num_subgraphs = _ref_num_subgraphs;
    targetDevice = _targetDevice;
    init_input_shapes(inputShape);
    auto f = ov::test::snippets::TwoInputsAndOutputsFunction(inputDynamicShapes);
    function = f.getOriginal();
    setIgnoreCallbackMode();
    abs_threshold = 5e-7;
}

void TwoInputsAndOutputsWithReversedOutputs::SetUp() {
    const auto& [inputShape, _ref_num_nodes, _ref_num_subgraphs, _targetDevice] = this->GetParam();
    ref_num_nodes = _ref_num_nodes;
    ref_num_subgraphs = _ref_num_subgraphs;
    targetDevice = _targetDevice;
    init_input_shapes(inputShape);
    auto f = ov::test::snippets::TwoInputsAndOutputsWithReversedOutputsFunction(inputDynamicShapes);
    function = f.getOriginal();
    setIgnoreCallbackMode();
    abs_threshold = 5e-7;
}

TEST_P(TwoInputsAndOutputs, CompareWithRefImpl) {
    run();
    validateNumSubgraphs();
}

TEST_P(TwoInputsAndOutputsWithReversedOutputs, CompareWithRefImpl) {
    run();
    validateNumSubgraphs();
}

} // namespace snippets
} // namespace test
} // namespace ov
