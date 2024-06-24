// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/common_utils.hpp"
#include "snippets/transpose_matmul.hpp"
#include "subgraph_matmul.hpp"
#include "functional_test_utils/skip_tests_config.hpp"

namespace ov {
namespace test {
namespace snippets {

std::string TransposeMatMul::getTestCaseName(testing::TestParamInfo<ov::test::snippets::TransposeMatMulParams> obj) {
    std::vector<InputShape> input_shapes;
    size_t transpose_position;
    std::vector<ov::element::Type> elem_types;
    std::string targetDevice;
    size_t num_nodes, num_subgraphs;
    std::tie(input_shapes, transpose_position, elem_types, num_nodes, num_subgraphs, targetDevice) = obj.param;
    std::ostringstream result;
    for (size_t i = 0; i < input_shapes.size(); ++i) {
        result << "IS[" << i << "]=" << input_shapes[i] << "_";
    }
    result << "Pos=" << transpose_position << "_";
    for (size_t i = 0; i < elem_types.size(); i++)
        result << "T[" << i <<"]=" << elem_types[i] << "_";
    result << "#N=" << num_nodes << "_";
    result << "#S=" << num_subgraphs << "_";
    result << "targetDevice=" << targetDevice;
    auto str = result.str();
    std::replace(str.begin(), str.end(), ',', '.');
    return str;
}

void TransposeMatMul::SetUp() {
    std::vector<InputShape> input_shapes;
    size_t transpose_position;
    std::vector<ov::element::Type> elem_types;
    std::tie(input_shapes, transpose_position, elem_types, ref_num_nodes, ref_num_subgraphs, targetDevice) = this->GetParam();
    init_input_shapes(input_shapes);

    auto f = ov::test::snippets::Transpose0213MatMulFunction(inputDynamicShapes, elem_types, transpose_position);
    function = f.getOriginal();
    if (!configuration.count("SNIPPETS_MODE")) {
        configuration.insert({"SNIPPETS_MODE", "IGNORE_CALLBACK"});
    }
}

void TransposeMatMulFQ::SetUp() {
    std::vector<InputShape> input_shapes;
    size_t transpose_position;
    std::vector<ov::element::Type> elem_types;
    std::tie(input_shapes, transpose_position, elem_types, ref_num_nodes, ref_num_subgraphs, targetDevice) = this->GetParam();
    init_input_shapes(input_shapes);

    auto f = ov::test::snippets::FQMatMulFunction(inputDynamicShapes, transpose_position);
    function = f.getOriginal();
    if (!configuration.count("SNIPPETS_MODE")) {
        configuration.insert({"SNIPPETS_MODE", "IGNORE_CALLBACK"});
    }
    abs_threshold = 5e-6;
}

void ExplicitTransposeMatMul::SetUp() {
    std::vector<InputShape> input_shapes;
    size_t transpose_position;
    std::vector<ov::element::Type> elem_types;
    std::tie(input_shapes, transpose_position, elem_types, ref_num_nodes, ref_num_subgraphs, targetDevice) = this->GetParam();
    init_input_shapes(input_shapes);

    auto f = ov::test::snippets::TransposeMatMulFunction(inputDynamicShapes);
    function = f.getOriginal();
    if (!configuration.count("SNIPPETS_MODE")) {
        configuration.insert({"SNIPPETS_MODE", "IGNORE_CALLBACK"});
    }
}

void ExplicitTransposeMatMulBias::SetUp() {
    std::vector<InputShape> input_shapes;
    size_t transpose_position;
    std::vector<ov::element::Type> elem_types;
    std::tie(input_shapes, transpose_position, elem_types, ref_num_nodes, ref_num_subgraphs, targetDevice) = this->GetParam();
    init_input_shapes(input_shapes);

    auto f = ov::test::snippets::TransposeMatMulBiasFunction(inputDynamicShapes);
    function = f.getOriginal();
    if (!configuration.count("SNIPPETS_MODE")) {
        configuration.insert({"SNIPPETS_MODE", "IGNORE_CALLBACK"});
    }
}

TEST_P(TransposeMatMul, CompareWithRefImpl) {
   SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
    validateNumSubgraphs();
}

TEST_P(TransposeMatMulFQ, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
    validateNumSubgraphs();
}

TEST_P(ExplicitTransposeMatMul, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
    validateNumSubgraphs();
}

TEST_P(ExplicitTransposeMatMulBias, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
    validateNumSubgraphs();
}

} // namespace snippets
} // namespace test
} // namespace ov
