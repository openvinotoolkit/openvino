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
    InputShape input_shape;
    ov::test::utils::ReductionType reduce_type;
    std::vector<int> axes;
    bool keep_dims;
    size_t num_nodes, num_subgraphs;
    std::string targetDevice;
    std::tie(input_shape, reduce_type, axes, keep_dims, num_nodes, num_subgraphs, targetDevice) = obj.param;

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
    InputShape input_shape;
    ov::test::utils::ReductionType reduce_type;
    std::vector<int> axes;
    bool keep_dims;
    std::tie(input_shape, reduce_type, axes, keep_dims, ref_num_nodes, ref_num_subgraphs, targetDevice) = this->GetParam();
    init_input_shapes({input_shape});

    auto f = ov::test::snippets::ReduceFunction(inputDynamicShapes, reduce_type, axes, keep_dims);
    function = f.getOriginal();

    if (!configuration.count("SNIPPETS_MODE")) {
        configuration.insert({"SNIPPETS_MODE", "IGNORE_CALLBACK"});
    }
}

TEST_P(Reduce, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
    validateNumSubgraphs();
}

} // namespace snippets
} // namespace test
} // namespace ov
