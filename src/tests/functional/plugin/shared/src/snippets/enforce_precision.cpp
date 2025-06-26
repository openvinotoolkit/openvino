// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/enforce_precision.hpp"

#include "common_test_utils/common_utils.hpp"
#include "subgraph_roll_matmul_roll.hpp"

namespace ov {
namespace test {
namespace snippets {

std::string EnforcePrecisionTest::getTestCaseName(testing::TestParamInfo<EnforcePrecisionTestParams> obj) {
    std::vector<ov::PartialShape> input_shapes;
    size_t num_nodes, num_subgraphs;
    std::string targetDevice;
    std::tie(input_shapes, num_nodes, num_subgraphs, targetDevice) = obj.param;

    std::ostringstream result;
    for (size_t i = 0; i < input_shapes.size(); ++i)
        result << "IS[" << i << "]=" << input_shapes[i] << "_";
    result << "#N=" << num_nodes << "_";
    result << "#S=" << num_subgraphs << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void EnforcePrecisionTest::SetUp() {
    std::vector<ov::PartialShape> input_shapes;
    std::tie(input_shapes, ref_num_nodes, ref_num_subgraphs, targetDevice) = this->GetParam();
    init_input_shapes(static_partial_shapes_to_test_representation(input_shapes));

    function = SubgraphRollMatMulRollFunction(input_shapes, ov::element::f32).getOriginal();

    setIgnoreCallbackMode();

    setInferenceType(element::bf16);
}

TEST_P(EnforcePrecisionTest, CompareWithRefImpl) {
    run();
    validateNumSubgraphs();
}

} // namespace snippets
} // namespace test
} // namespace ov
