// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/common_utils.hpp"
#include "snippets/transpose_softmax.hpp"
#include "subgraph_softmax.hpp"
#include "functional_test_utils/skip_tests_config.hpp"

namespace ov {
namespace test {
namespace snippets {

std::string TransposeSoftmax::getTestCaseName(testing::TestParamInfo<ov::test::snippets::TransposeSoftmaxParams> obj) {
    const auto& [inputShapes, order, axis, num_nodes, num_subgraphs, targetDevice] = obj.param;

    std::ostringstream result;
    for (size_t i = 0; i < inputShapes.size(); ++i) {
        result << "IS[" << i<< "]=" << ov::test::utils::partialShape2str({inputShapes[i].first}) << "_";
        result << "TS[" << i<< "]=";
        for (const auto& shape : inputShapes[i].second) {
            result << "(" << ov::test::utils::vec2str(shape) << ")_";
        }
    }
    result << "TO=" << ov::test::utils::vec2str(order) << "_";
    result << "Axis=" << axis << "_";
    result << "#N=" << num_nodes << "_";
    result << "#S=" << num_subgraphs << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void TransposeSoftmax::SetUp() {
    const auto& [inputShapes, order, axis, _ref_num_nodes, _ref_num_subgraphs, _targetDevice] = this->GetParam();
    ref_num_nodes = _ref_num_nodes;
    ref_num_subgraphs = _ref_num_subgraphs;
    targetDevice = _targetDevice;
    init_input_shapes(inputShapes);

    auto f = ov::test::snippets::TransposeSoftmaxFunction(inputDynamicShapes, order, axis);
    function = f.getOriginal();

    setIgnoreCallbackMode();
}

void TransposeSoftmaxEltwise::SetUp() {
    const auto& [inputShapes, order, axis, _ref_num_nodes, _ref_num_subgraphs, _targetDevice] = this->GetParam();
    ref_num_nodes = _ref_num_nodes;
    ref_num_subgraphs = _ref_num_subgraphs;
    targetDevice = _targetDevice;
    init_input_shapes(inputShapes);

    auto f = ov::test::snippets::TransposeSoftmaxEltwiseFunction(inputDynamicShapes, order, axis);
    function = f.getOriginal();

    setIgnoreCallbackMode();

    abs_threshold = 1e-6;
}

TEST_P(TransposeSoftmax, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
    validateNumSubgraphs();
}

TEST_P(TransposeSoftmaxEltwise, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
    validateNumSubgraphs();
}


} // namespace snippets
} // namespace test
} // namespace ov
