// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/transpose_matmul.hpp"

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "snippets/matmul.hpp"
#include "subgraph_matmul.hpp"

namespace ov {
namespace test {
namespace snippets {

std::string TransposeMatMul::getTestCaseName(testing::TestParamInfo<ov::test::snippets::TransposeMatMulParams> obj) {
    std::vector<InputShape> input_shapes;
    size_t transpose_position;
    std::vector<ov::element::Type> elem_types;
    MatMulType matmul_type;
    std::string targetDevice;
    size_t num_nodes, num_subgraphs;
    std::tie(input_shapes, transpose_position, elem_types, matmul_type, num_nodes, num_subgraphs, targetDevice) = obj.param;
    std::ostringstream result;
    for (size_t i = 0; i < input_shapes.size(); ++i) {
        result << "IS[" << i << "]=" << input_shapes[i] << "_";
    }
    result << "Pos=" << transpose_position << "_";
    for (size_t i = 0; i < elem_types.size(); i++)
        result << "T[" << i <<"]=" << elem_types[i] << "_";
    result << matmul_type << "_";
    result << "#N=" << num_nodes << "_";
    result << "#S=" << num_subgraphs << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void TransposeMatMul::SetUp() {
    std::vector<InputShape> input_shapes;
    std::vector<ov::element::Type> elem_types;
    std::tie(input_shapes, transpose_position, elem_types, matmul_type, ref_num_nodes, ref_num_subgraphs, targetDevice) = this->GetParam();
    init_input_shapes(input_shapes);

    const auto builder = get_builder(elem_types);
    function = builder->getOriginal();
    filter_shape_info(builder->get_constant_input_idces());
    if (!configuration.count("SNIPPETS_MODE")) {
        configuration.insert({"SNIPPETS_MODE", "IGNORE_CALLBACK"});
    }
}

std::shared_ptr<MatMulFunctionBase> TransposeMatMul::get_builder(const std::vector<ov::element::Type>& types) {
    return std::make_shared<Transpose0213MatMulFunction>(inputDynamicShapes, types, matmul_type, transpose_position);
}

std::shared_ptr<MatMulFunctionBase> TransposeMatMulFQ::get_builder(const std::vector<ov::element::Type>& types) {
    return std::make_shared<FQMatMulFunction>(inputDynamicShapes, matmul_type, transpose_position);
}

std::shared_ptr<MatMulFunctionBase> ExplicitTransposeMatMul::get_builder(const std::vector<ov::element::Type>& types) {
    return std::make_shared<TransposeMatMulFunction>(inputDynamicShapes);
}

std::shared_ptr<MatMulFunctionBase> ExplicitTransposeMatMulBias::get_builder(const std::vector<ov::element::Type>& types) {
    return std::make_shared<TransposeMatMulBiasFunction>(inputDynamicShapes);
}

TEST_P(TransposeMatMul, CompareWithRefImpl) {
   SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
    validateNumSubgraphs();
}

TEST_P(TransposeMatMulFQ, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    abs_threshold = 5e-6;
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
