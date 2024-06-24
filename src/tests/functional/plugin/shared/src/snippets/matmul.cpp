// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/common_utils.hpp"
#include "snippets/matmul.hpp"
#include "subgraph_matmul.hpp"
#include "functional_test_utils/skip_tests_config.hpp"

namespace ov {
namespace test {
namespace snippets {

std::string MatMul::getTestCaseName(testing::TestParamInfo<ov::test::snippets::MatMulParams> obj) {
    std::vector<ov::test::InputShape> input_shapes;
    std::vector<ov::element::Type> elem_types;
    std::string targetDevice;
    size_t num_nodes, num_subgraphs;
    std::tie(input_shapes, elem_types, num_nodes, num_subgraphs, targetDevice) = obj.param;
    std::ostringstream result;
    for (size_t i = 0; i < input_shapes.size(); i++)
        result << "IS[" << i << "]=" << input_shapes[i] << "_";

    for (size_t i = 0; i < elem_types.size(); i++)
        result << "T[" << i <<"]=" << elem_types[i] << "_";
    result << "#N=" << num_nodes << "_";
    result << "#S=" << num_subgraphs << "_";
    result << "targetDevice=" << targetDevice;
    auto str = result.str();
    std::replace(str.begin(), str.end(), ',', '.');
    return str;
}

void MatMul::SetUp() {
    std::vector<ov::test::InputShape> input_shapes;
    std::vector<ov::element::Type> elem_types;
    std::tie(input_shapes, elem_types, ref_num_nodes, ref_num_subgraphs, targetDevice) = this->GetParam();
    init_input_shapes(input_shapes);

    init_subgraph(elem_types);
    if (!configuration.count("SNIPPETS_MODE")) {
        configuration.insert({"SNIPPETS_MODE", "IGNORE_CALLBACK"});
    }
}

void MatMul::init_subgraph(const std::vector<ov::element::Type>& types) {
    auto f = ov::test::snippets::MatMulFunction(inputDynamicShapes, types);
    function = f.getOriginal();
}

void MatMulFQ::init_subgraph(const std::vector<ov::element::Type>& types) {
    auto f = ov::test::snippets::FQMatMulFunction(inputDynamicShapes);
    function = f.getOriginal();
}

void MatMulBias::init_subgraph(const std::vector<ov::element::Type>& types) {
    auto f = ov::test::snippets::MatMulBiasFunction(inputDynamicShapes, types);
    function = f.getOriginal();
}

void MatMulBiasQuantized::init_subgraph(const std::vector<ov::element::Type>& types) {
    auto f = ov::test::snippets::MatMulBiasQuantizedFunction(inputDynamicShapes, types);
    function = f.getOriginal();
}

void MatMulQuantized::init_subgraph(const std::vector<ov::element::Type>& types) {
    auto f = ov::test::snippets::MatMulsQuantizedFunction(inputDynamicShapes, types);
    function = f.getOriginal();
}

void MatMulQuantizedSoftmax::init_subgraph(const std::vector<ov::element::Type>& types) {
    auto f = ov::test::snippets::MatMulsQuantizedSoftmaxFunction(inputDynamicShapes, types);
    function = f.getOriginal();
}

TEST_P(MatMul, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
    validateNumSubgraphs();
}

TEST_P(MatMulFQ, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    abs_threshold = 0.5;
    run();
    validateNumSubgraphs();
}

TEST_P(MatMulBias, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
    validateNumSubgraphs();
}

TEST_P(MatMulBiasQuantized, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
    validateNumSubgraphs();
}

TEST_P(MatMulQuantized, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
    validateNumSubgraphs();
}

TEST_P(MatMulQuantizedSoftmax, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    abs_threshold = 4e-6;
    run();
    validateNumSubgraphs();
}

} // namespace snippets
} // namespace test
} // namespace ov
