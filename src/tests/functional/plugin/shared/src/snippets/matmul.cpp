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
    ov::element::Type elem_type;
    std::string targetDevice;
    size_t num_nodes, num_subgraphs;
    std::tie(input_shapes, elem_type, num_nodes, num_subgraphs, targetDevice) = obj.param;
    std::ostringstream result;
    for (size_t i = 0; i < input_shapes.size(); i++)
        result << "IS[" << i <<"]=" << CommonTestUtils::partialShape2str({input_shapes[i]}) << "_";
    result << "T=" << elem_type << "_";
    result << "#N=" << num_nodes << "_";
    result << "#S=" << num_subgraphs << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void MatMul::SetUp() {
    std::vector<ov::PartialShape> input_shapes;
    ov::element::Type elem_type;
    std::tie(input_shapes, elem_type, ref_num_nodes, ref_num_subgraphs, targetDevice) = this->GetParam();
    init_input_shapes(dynamic_shapes_to_test_representation(input_shapes));

    auto f = ov::test::snippets::MatMulSinhFunction(input_shapes);
    function = f.getOriginal();
    if (!configuration.count(InferenceEngine::PluginConfigInternalParams::KEY_SNIPPETS_MODE)) {
        configuration.insert({InferenceEngine::PluginConfigInternalParams::KEY_SNIPPETS_MODE,
                              InferenceEngine::PluginConfigInternalParams::IGNORE_CALLBACK});
    }
}

void MatMulBias::SetUp() {
    std::vector<ov::PartialShape> input_shapes;
    ov::element::Type elem_type;
    std::tie(input_shapes, elem_type, ref_num_nodes, ref_num_subgraphs, targetDevice) = this->GetParam();
    init_input_shapes(dynamic_shapes_to_test_representation(input_shapes));

    auto f = ov::test::snippets::MatMulBiasSinhFunction(input_shapes);
    function = f.getOriginal();
    if (!configuration.count(InferenceEngine::PluginConfigInternalParams::KEY_SNIPPETS_MODE)) {
        configuration.insert({InferenceEngine::PluginConfigInternalParams::KEY_SNIPPETS_MODE,
                              InferenceEngine::PluginConfigInternalParams::IGNORE_CALLBACK});
    }
}

void ExplicitTransposeMatMul::SetUp() {
    std::vector<ov::PartialShape> input_shapes;
    ov::element::Type elem_type;
    std::tie(input_shapes, elem_type, ref_num_nodes, ref_num_subgraphs, targetDevice) = this->GetParam();
    init_input_shapes(dynamic_shapes_to_test_representation(input_shapes));

    auto f = ov::test::snippets::TransposeMatMulSinhFunction(input_shapes);
    function = f.getOriginal();
    if (!configuration.count(InferenceEngine::PluginConfigInternalParams::KEY_SNIPPETS_MODE)) {
        configuration.insert({InferenceEngine::PluginConfigInternalParams::KEY_SNIPPETS_MODE,
                              InferenceEngine::PluginConfigInternalParams::IGNORE_CALLBACK});
    }
}

void ExplicitTransposeMatMulBias::SetUp() {
    std::vector<ov::PartialShape> input_shapes;
    ov::element::Type elem_type;
    std::tie(input_shapes, elem_type, ref_num_nodes, ref_num_subgraphs, targetDevice) = this->GetParam();
    init_input_shapes(dynamic_shapes_to_test_representation(input_shapes));

    auto f = ov::test::snippets::TransposeMatMulBiasSinhFunction(input_shapes);
    function = f.getOriginal();
    if (!configuration.count(InferenceEngine::PluginConfigInternalParams::KEY_SNIPPETS_MODE)) {
        configuration.insert({InferenceEngine::PluginConfigInternalParams::KEY_SNIPPETS_MODE,
                              InferenceEngine::PluginConfigInternalParams::IGNORE_CALLBACK});
    }
}

void ExplicitTransposeMulMatMulBias::SetUp() {
    std::vector<ov::PartialShape> input_shapes;
    ov::element::Type elem_type;
    std::tie(input_shapes, elem_type, ref_num_nodes, ref_num_subgraphs, targetDevice) = this->GetParam();
    init_input_shapes(dynamic_shapes_to_test_representation(input_shapes));

    auto f = ov::test::snippets::TransposeMulMatMulBiasSinhFunction(input_shapes);
    function = f.getOriginal();
    if (!configuration.count(InferenceEngine::PluginConfigInternalParams::KEY_SNIPPETS_MODE)) {
        configuration.insert({InferenceEngine::PluginConfigInternalParams::KEY_SNIPPETS_MODE,
                              InferenceEngine::PluginConfigInternalParams::IGNORE_CALLBACK});
    }
}

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
    init_input_shapes(dynamic_shapes_to_test_representation(input_shapes));

    auto f = ov::test::snippets::Transpose0213MatMulSinhFunction(input_shapes, transpose_position);
    function = f.getOriginal();
    if (!configuration.count(InferenceEngine::PluginConfigInternalParams::KEY_SNIPPETS_MODE)) {
        configuration.insert({InferenceEngine::PluginConfigInternalParams::KEY_SNIPPETS_MODE,
                              InferenceEngine::PluginConfigInternalParams::IGNORE_CALLBACK});
    }
}

TEST_P(MatMulBias, CompareWithRefImpl) {
    run();
    validateNumSubgraphs();
}

TEST_P(ExplicitTransposeMatMul, CompareWithRefImpl) {
    run();
    validateNumSubgraphs();
}

TEST_P(ExplicitTransposeMatMulBias, CompareWithRefImpl) {
    run();
    validateNumSubgraphs();
}

TEST_P(ExplicitTransposeMulMatMulBias, CompareWithRefImpl) {
    run();
    validateNumSubgraphs();
}

TEST_P(TransposeMatMul, CompareWithRefImpl) {
    run();
    validateNumSubgraphs();
}

} // namespace snippets
} // namespace test
} // namespace ov
