// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/common_utils.hpp"
#include "snippets/transpose.hpp"
#include "subgraph_transpose.hpp"
#include "functional_test_utils/skip_tests_config.hpp"

namespace ov {
namespace test {
namespace snippets {

std::string Transpose::getTestCaseName(testing::TestParamInfo<ov::test::snippets::TransposeParams> obj) {
    const auto& [inputShapes, order, num_nodes, num_subgraphs, targetDevice] = obj.param;

    std::ostringstream result;
    result << "IS=" << inputShapes << "_";
    result << "Order=" << ov::test::utils::vec2str(order) << "_";
    result << "#N=" << num_nodes << "_";
    result << "#S=" << num_subgraphs << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void Transpose::SetUp() {
    const auto& [inputShape, order, _ref_num_nodes, _ref_num_subgraphs, _targetDevice] = this->GetParam();
    ref_num_nodes = _ref_num_nodes;
    ref_num_subgraphs = _ref_num_subgraphs;
    targetDevice = _targetDevice;
    init_input_shapes({inputShape});

    auto f = ov::test::snippets::TransposeFunction(inputDynamicShapes, order);
    function = f.getOriginal();
    setIgnoreCallbackMode();
}

std::string TransposeMul::getTestCaseName(testing::TestParamInfo<ov::test::snippets::TransposeMulParams> obj) {
    const auto& [inputShapes, order, num_nodes, num_subgraphs, targetDevice] = obj.param;

    std::ostringstream result;
    result << "IS[0]=" << inputShapes.first << "_";
    result << "IS[1]=" << inputShapes.second << "_";
    result << "Order=" << ov::test::utils::vec2str(order) << "_";
    result << "#N=" << num_nodes << "_";
    result << "#S=" << num_subgraphs << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void TransposeMul::SetUp() {
    const auto& [inputShapes, order, _ref_num_nodes, _ref_num_subgraphs, _targetDevice] = this->GetParam();
    ref_num_nodes = _ref_num_nodes;
    ref_num_subgraphs = _ref_num_subgraphs;
    targetDevice = _targetDevice;
    init_input_shapes({inputShapes.first, inputShapes.second});
    auto f = ov::test::snippets::TransposeMulFunction(inputDynamicShapes, order);
    function = f.getOriginal();
    setIgnoreCallbackMode();
}

TEST_P(Transpose, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
    validateNumSubgraphs();
}

TEST_P(TransposeMul, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
    validateNumSubgraphs();
}

} // namespace snippets
} // namespace test
} // namespace ov
