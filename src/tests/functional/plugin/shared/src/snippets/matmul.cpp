// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/common_utils.hpp"
#include "snippets/matmul.hpp"
#include "subgraph_matmul.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "cpp_interfaces/interface/ie_internal_plugin_config.hpp"

namespace ov {
namespace test {
namespace snippets {

std::string MatMul::getTestCaseName(testing::TestParamInfo<ov::test::snippets::MatMulParams> obj) {
    std::vector<ov::PartialShape> input_shapes;
    std::vector<ov::element::Type> elem_types;
    std::string targetDevice;
    size_t num_nodes, num_subgraphs;
    std::tie(input_shapes, elem_types, num_nodes, num_subgraphs, targetDevice) = obj.param;
    std::ostringstream result;
    for (size_t i = 0; i < input_shapes.size(); i++)
        result << "IS[" << i <<"]=" << ov::test::utils::partialShape2str({input_shapes[i]}) << "_";
    for (size_t i = 0; i < elem_types.size(); i++)
        result << "T[" << i <<"]=" << elem_types[i] << "_";
    result << "#N=" << num_nodes << "_";
    result << "#S=" << num_subgraphs << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void MatMul::SetUp() {
    std::vector<ov::PartialShape> input_shapes;
    std::vector<ov::element::Type> elem_types;
    std::tie(input_shapes, elem_types, ref_num_nodes, ref_num_subgraphs, targetDevice) = this->GetParam();
    init_input_shapes(static_partial_shapes_to_test_representation(input_shapes));

    init_subgraph(input_shapes, elem_types);
    if (!configuration.count(InferenceEngine::PluginConfigInternalParams::KEY_SNIPPETS_MODE)) {
        configuration.insert({InferenceEngine::PluginConfigInternalParams::KEY_SNIPPETS_MODE,
                              InferenceEngine::PluginConfigInternalParams::IGNORE_CALLBACK});
    }
}

void MatMul::init_subgraph(const std::vector<PartialShape>& inputShapes, const std::vector<ov::element::Type>& types) {
    auto f = ov::test::snippets::MatMulFunction(inputShapes, types);
    function = f.getOriginal();
}

void MatMulFQ::init_subgraph(const std::vector<PartialShape>& inputShapes, const std::vector<ov::element::Type>& types) {
    auto f = ov::test::snippets::FQMatMulFunction(inputShapes);
    function = f.getOriginal();
}

void MatMulBias::init_subgraph(const std::vector<PartialShape>& inputShapes, const std::vector<ov::element::Type>& types) {
    auto f = ov::test::snippets::MatMulBiasFunction(inputShapes, types);
    function = f.getOriginal();
}

void MatMulBiasQuantized::init_subgraph(const std::vector<PartialShape>& inputShapes, const std::vector<ov::element::Type>& types) {
    auto f = ov::test::snippets::MatMulBiasQuantizedFunction(inputShapes, types);
    function = f.getOriginal();
}

void MatMulsQuantized::init_subgraph(const std::vector<PartialShape>& inputShapes, const std::vector<ov::element::Type>& types) {
    auto f = ov::test::snippets::MatMulsQuantizedFunction(inputShapes, types);
    function = f.getOriginal();
}

void MatMulsQuantizedSoftmax::init_subgraph(const std::vector<PartialShape>& inputShapes, const std::vector<ov::element::Type>& types) {
    auto f = ov::test::snippets::MatMulsQuantizedSoftmaxFunction(inputShapes, types);
    function = f.getOriginal();
}

TEST_P(MatMul, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
    validateNumSubgraphs();
}

TEST_P(MatMulFQ, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
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

TEST_P(MatMulsQuantized, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
    validateNumSubgraphs();
}

TEST_P(MatMulsQuantizedSoftmax, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
    validateNumSubgraphs();
}

} // namespace snippets
} // namespace test
} // namespace ov
