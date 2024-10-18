// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/matmul.hpp"

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "subgraph_matmul.hpp"

namespace ov {
namespace test {
namespace snippets {

void MatMulBase::filter_shape_info(const std::set<size_t>& idces_to_remove) {
    for (auto idx_it = idces_to_remove.rbegin(); idx_it != idces_to_remove.rend(); ++idx_it) {
        const auto& idx = * idx_it;
        OPENVINO_ASSERT(idx < inputDynamicShapes.size());
        inputDynamicShapes.erase(inputDynamicShapes.begin() + idx);
        for (auto& target_shapes : targetStaticShapes) {
            OPENVINO_ASSERT(idx < target_shapes.size());
            target_shapes.erase(target_shapes.begin() + idx);
        }
    }
}

std::string MatMul::getTestCaseName(testing::TestParamInfo<ov::test::snippets::MatMulParams> obj) {
    std::vector<ov::test::InputShape> input_shapes;
    std::vector<ov::element::Type> elem_types;
    MatMulType mm_type;
    std::string targetDevice;
    size_t num_nodes, num_subgraphs;
    std::tie(input_shapes, elem_types, mm_type, num_nodes, num_subgraphs, targetDevice) = obj.param;
    std::ostringstream result;
    for (size_t i = 0; i < input_shapes.size(); i++)
        result << "IS[" << i << "]=" << input_shapes[i] << "_";

    for (size_t i = 0; i < elem_types.size(); i++)
        result << "T[" << i <<"]=" << elem_types[i] << "_";

    result << mm_type << "_";
    result << "#N=" << num_nodes << "_";
    result << "#S=" << num_subgraphs << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void MatMul::SetUp() {
    std::vector<ov::test::InputShape> input_shapes;
    std::vector<ov::element::Type> elem_types;
    std::tie(input_shapes, elem_types, matmul_type, ref_num_nodes, ref_num_subgraphs, targetDevice) = this->GetParam();
    init_input_shapes(input_shapes);

    const auto builder = get_builder(elem_types);
    function = builder->getOriginal();
    filter_shape_info(builder->get_constant_input_idces());
    if (!configuration.count("SNIPPETS_MODE")) {
        configuration.insert({"SNIPPETS_MODE", "IGNORE_CALLBACK"});
    }
}

std::shared_ptr<MatMulFunctionBase> MatMul::get_builder(const std::vector<ov::element::Type>& types) {
    return std::make_shared<MatMulFunction>(inputDynamicShapes, types, matmul_type);
}

std::shared_ptr<MatMulFunctionBase> MatMulTransposeB::get_builder(const std::vector<ov::element::Type>& types) {
    return std::make_shared<MatMulFunction>(inputDynamicShapes, types, matmul_type, true);
}

std::shared_ptr<MatMulFunctionBase> MatMulFQ::get_builder(const std::vector<ov::element::Type>& types) {
    return std::make_shared<FQMatMulFunction>(inputDynamicShapes, matmul_type);
}

std::shared_ptr<MatMulFunctionBase> MatMulBias::get_builder(const std::vector<ov::element::Type>& types) {
    return std::make_shared<MatMulBiasFunction>(inputDynamicShapes, types, matmul_type);
}

std::shared_ptr<MatMulFunctionBase> MatMulBiasQuantized::get_builder(const std::vector<ov::element::Type>& types) {
    return std::make_shared<MatMulBiasQuantizedFunction>(inputDynamicShapes, types, matmul_type);
}

std::shared_ptr<MatMulFunctionBase> MatMulsQuantized::get_builder(const std::vector<ov::element::Type>& types) {
    return std::make_shared<MatMulsQuantizedFunction>(inputDynamicShapes, types, matmul_type);
}

std::shared_ptr<MatMulFunctionBase> MatMulsQuantizedSoftmax::get_builder(const std::vector<ov::element::Type>& types) {
    return std::make_shared<MatMulsQuantizedSoftmaxFunction>(inputDynamicShapes, types, matmul_type);
}

std::shared_ptr<MatMulFunctionBase> MatMulEltwiseChain::get_builder(const std::vector<ov::element::Type>& types) {
    return std::make_shared<MatMulEltwiseChainFunction>(inputDynamicShapes, types, matmul_type);
}

std::shared_ptr<MatMulFunctionBase> MatMulEltwiseChainCascade::get_builder(const std::vector<ov::element::Type>& types) {
    return std::make_shared<MatMulEltwiseChainCascadeFunction>(inputDynamicShapes, types, matmul_type);
}

TEST_P(MatMul, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
    validateNumSubgraphs();
}

TEST_P(MatMulTransposeB, CompareWithRefImpl) {
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

TEST_P(MatMulsQuantized, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
    validateNumSubgraphs();
}

TEST_P(MatMulsQuantizedSoftmax, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    abs_threshold = 4e-6;
    run();
    validateNumSubgraphs();
}

TEST_P(MatMulEltwiseChain, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
    validateNumSubgraphs();
}

TEST_P(MatMulEltwiseChainCascade, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
    validateNumSubgraphs();
}

} // namespace snippets
} // namespace test
} // namespace ov
