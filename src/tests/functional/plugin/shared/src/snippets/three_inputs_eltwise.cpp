// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/common_utils.hpp"
#include "snippets/three_inputs_eltwise.hpp"
#include "subgraph_simple.hpp"
#include "functional_test_utils/skip_tests_config.hpp"

namespace ov {
namespace test {
namespace snippets {

std::string ThreeInputsEltwise::getTestCaseName(const testing::TestParamInfo<ov::test::snippets::ThreeInputsEltwiseParams>& obj) {
    const auto& [inputShapes0, inputShapes1, inputShapes2, num_nodes, num_subgraphs, targetDevice] = obj.param;

    std::ostringstream result;
    result << "IS[0]=" << ov::test::utils::partialShape2str({inputShapes0.first}) << "_";
    result << "TS[0]=";
    for (const auto& shape : inputShapes0.second) {
        result << "(" << ov::test::utils::vec2str(shape) << ")_";
    }
    result << "IS[1]=" << ov::test::utils::partialShape2str({inputShapes1.first}) << "_";
    result << "TS[1]=";
    for (const auto& shape : inputShapes1.second) {
        result << "(" << ov::test::utils::vec2str(shape) << ")_";
    }
    result << "IS[2]=" << ov::test::utils::partialShape2str({inputShapes2.first}) << "_";
    result << "TS[2]=";
    for (const auto& shape : inputShapes2.second) {
        result << "(" << ov::test::utils::vec2str(shape) << ")_";
    }
    result << "#N=" << num_nodes << "_";
    result << "#S=" << num_subgraphs << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void ThreeInputsEltwise::SetUp() {
    const auto& [inputShape0, inputShape1, inputShape2, _ref_num_nodes, _ref_num_subgraphs, _targetDevice] =
        this->GetParam();
    ref_num_nodes = _ref_num_nodes;
    ref_num_subgraphs = _ref_num_subgraphs;
    targetDevice = _targetDevice;
    init_input_shapes({inputShape0, inputShape1, inputShape2});

    auto f = ov::test::snippets::EltwiseThreeInputsFunction(inputDynamicShapes);
    function = f.getOriginal();
    setIgnoreCallbackMode();
}

TEST_P(ThreeInputsEltwise, CompareWithRefImpl) {
    run();
    validateNumSubgraphs();
}

} // namespace snippets
} // namespace test
} // namespace ov
