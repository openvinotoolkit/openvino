// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/reduce.hpp"

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "subgraph_reduce.hpp"

namespace ov {
namespace test {
namespace snippets {

std::string Reduce::getTestCaseName(testing::TestParamInfo<ov::test::snippets::ReduceParams> obj) {
    const auto& [input_shape, reduce_type, axes, keep_dims, num_nodes, num_subgraphs, targetDevice] = obj.param;

    std::ostringstream result;
    result << "IS=" << ov::test::utils::partialShape2str({input_shape.first}) << "_";
    result << "TS=";
    for (const auto& shape : input_shape.second) {
        result << "(" << ov::test::utils::vec2str(shape) << ")_";
    }
    result << "reduce_type=" << reduce_type << "_";
    result << "Axes=" << ov::test::utils::vec2str(axes) << "_";
    result << "#N=" << num_nodes << "_";
    result << "#S=" << num_subgraphs << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void Reduce::SetUp() {
    const auto& [input_shape, reduce_type, axes, keep_dims, _ref_num_nodes, _ref_num_subgraphs, _targetDevice] =
        this->GetParam();
    ref_num_nodes = _ref_num_nodes;
    ref_num_subgraphs = _ref_num_subgraphs;
    targetDevice = _targetDevice;
    init_input_shapes({input_shape});

    auto f = ov::test::snippets::ReduceFunction(inputDynamicShapes, reduce_type, axes, keep_dims);
    function = f.getOriginal();

    setIgnoreCallbackMode();
}

TEST_P(Reduce, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
    validateNumSubgraphs();
}

} // namespace snippets
} // namespace test
} // namespace ov
