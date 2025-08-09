// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/matmul.hpp"

#include "common_test_utils/common_utils.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>
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
    auto [input_shapes, elem_types, mm_type, num_nodes, num_subgraphs, targetDevice, additional_config] = obj.param;
    std::ostringstream result;
    for (size_t i = 0; i < input_shapes.size(); i++)
        result << "IS[" << i << "]=" << input_shapes[i] << "_";

    for (size_t i = 0; i < elem_types.size(); i++)
        result << "T[" << i <<"]=" << elem_types[i] << "_";

    result << mm_type << "_";
    result << "#N=" << num_nodes << "_";
    result << "#S=" << num_subgraphs << "_";
    result << "targetDevice=" << targetDevice;
    if (!additional_config.empty()) {
        result << "_PluginConf";
        for (auto& item : additional_config) {
            result << "_" << item.first << "=" << item.second.as<std::string>();
        }
    }
    return result.str();
}

void MatMulBase::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    inputs.clear();
    const auto& model_inputs = function->inputs();

    for (int i = 0; i < model_inputs.size(); ++i) {
        const auto& model_input = model_inputs[i];
        ov::Tensor tensor;
        ov::test::utils::InputGenerateData in_data;
        // To avoid big relative errors in the vicinity of zero, only positive values are generated for bf16 precision
        in_data.start_from = model_input.get_element_type() == ov::element::bf16 ? 0 : -1;
        in_data.range = 2;
        in_data.resolution = 256;
        tensor =
            ov::test::utils::create_and_fill_tensor(model_input.get_element_type(), targetInputStaticShapes[i], in_data);
        inputs.insert({model_input.get_node_shared_ptr(), tensor});
    }
}


void MatMul::SetUp() {
    const auto& [input_shapes,
                 elem_types,
                 _matmul_type,
                 _ref_num_nodes,
                 _ref_num_subgraphs,
                 _targetDevice,
                 additional_config] = this->GetParam();
    matmul_type = _matmul_type;
    ref_num_nodes = _ref_num_nodes;
    ref_num_subgraphs = _ref_num_subgraphs;
    targetDevice = _targetDevice;
    init_input_shapes(input_shapes);

    const auto builder = get_builder(elem_types);
    function = builder->getOriginal();
    filter_shape_info(builder->get_constant_input_idces());
    configuration.insert(additional_config.begin(), additional_config.end());
    setIgnoreCallbackMode();
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
