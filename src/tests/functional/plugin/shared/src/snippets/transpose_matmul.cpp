// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/common_utils.hpp"
#include "snippets/transpose_matmul.hpp"
#include "subgraph_matmul.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "cpp_interfaces/interface/ie_internal_plugin_config.hpp"

namespace ov {
namespace test {
namespace snippets {

std::string TransposeMatMul::getTestCaseName(testing::TestParamInfo<ov::test::snippets::TransposeMatMulParams> obj) {
    std::vector<ov::PartialShape> input_shapes;
    size_t transpose_position;
    ov::element::Type elem_type;
    std::string targetDevice;
    size_t num_nodes, num_subgraphs;
    std::tie(input_shapes, transpose_position, elem_type, num_nodes, num_subgraphs, targetDevice) = obj.param;
    if (input_shapes.size() != 2)
        IE_THROW() << "Invalid input shapes vector size";
    std::ostringstream result;
    result << "IS[0]=" << CommonTestUtils::partialShape2str({input_shapes[0]}) << "_";
    result << "IS[1]=" << CommonTestUtils::partialShape2str({input_shapes[1]}) << "_";
    result << "Pos=" << transpose_position << "_";
    result << "T=" << elem_type << "_";
    result << "#N=" << num_nodes << "_";
    result << "#S=" << num_subgraphs << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void TransposeMatMul::SetUp() {
    std::vector<ov::PartialShape> input_shapes;
    size_t transpose_position;
    ov::element::Type elem_type;
    std::tie(input_shapes, transpose_position, elem_type, ref_num_nodes, ref_num_subgraphs, targetDevice) = this->GetParam();
    init_input_shapes(static_partial_shapes_to_test_representation(input_shapes));

    auto f = ov::test::snippets::Transpose0213MatMulFunction(input_shapes, transpose_position);
    function = f.getOriginal();
    if (!configuration.count(InferenceEngine::PluginConfigInternalParams::KEY_SNIPPETS_MODE)) {
        configuration.insert({InferenceEngine::PluginConfigInternalParams::KEY_SNIPPETS_MODE,
                              InferenceEngine::PluginConfigInternalParams::IGNORE_CALLBACK});
    }
}

TEST_P(TransposeMatMul, CompareWithRefImpl) {
    run();
    validateNumSubgraphs();
}

} // namespace snippets
} // namespace test
} // namespace ov
