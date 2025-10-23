// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/common_utils.hpp"
#include "snippets/add.hpp"
#include "subgraph_simple.hpp"
#include "functional_test_utils/skip_tests_config.hpp"

namespace ov {
namespace test {
namespace snippets {

std::string Add::getTestCaseName(const testing::TestParamInfo<ov::test::snippets::AddParams>& obj) {
    const auto& [inputShapes0, inputShapes1, type, num_nodes, num_subgraphs, targetDevice] = obj.param;

    std::ostringstream result;
    result << "IS[0]=" << ov::test::utils::partialShape2str({inputShapes0.first}) << "_";
    result << "TS[0]=";
    for (const auto& shape : inputShapes0.second) {
        result << "(" << ov::test::utils::vec2str(shape) << ")_";
    }
    result << "IS[1]=" << ov::test::utils::partialShape2str({inputShapes1.first}) << "_";
    result << "TS[1]=";
    for (const auto& shape : inputShapes1.second) {
        result << "(" << ov::test::utils::vec2str(shape) << ")_";
    }
    result << "T=" << type << "_";
    result << "#N=" << num_nodes << "_";
    result << "#S=" << num_subgraphs << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void Add::SetUp() {
    const auto& [inputShape0, inputShape1, type, _ref_num_nodes, _ref_num_subgraphs, _targetDevice] = this->GetParam();
    ref_num_nodes = _ref_num_nodes;
    ref_num_subgraphs = _ref_num_subgraphs;
    targetDevice = _targetDevice;
    init_input_shapes({inputShape0, inputShape1});
    auto f = ov::test::snippets::AddFunction(inputDynamicShapes);
    function = f.getOriginal();
    setInferenceType(type);
    setIgnoreCallbackMode();
}

std::string AddConst::getTestCaseName(const testing::TestParamInfo<ov::test::snippets::AddConstParams>& obj) {
    const auto& [inputShapes, constShape, type, num_nodes, num_subgraphs, targetDevice] = obj.param;

    std::ostringstream result;
    result << "IS[0]=" << ov::test::utils::partialShape2str({inputShapes.first}) << "_";
    result << "TS[0]=";
    for (const auto& shape : inputShapes.second) {
        result << "(" << ov::test::utils::vec2str(shape) << ")_";
    }
    result << "IS_ConstShape=" << ov::test::utils::partialShape2str({constShape}) << "_";
    result << "T=" << type << "_";
    result << "#N=" << num_nodes << "_";
    result << "#S=" << num_subgraphs << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void AddConst::SetUp() {
    const auto& [inputShape, constShape, type, _ref_num_nodes, _ref_num_subgraphs, _targetDevice] = this->GetParam();
    ref_num_nodes = _ref_num_nodes;
    ref_num_subgraphs = _ref_num_subgraphs;
    targetDevice = _targetDevice;
    init_input_shapes({{inputShape}});
    auto f = ov::test::snippets::AddConstFunction({inputDynamicShapes}, constShape);
    function = f.getOriginal();
    setInferenceType(type);
    setIgnoreCallbackMode();
    if (type == ov::element::f16) {
        abs_threshold = 3e-2;
    }
}

void AddRollConst::SetUp() {
    const auto& [inputShape, constShape, type, _ref_num_nodes, _ref_num_subgraphs, _targetDevice] = this->GetParam();
    ref_num_nodes = _ref_num_nodes;
    ref_num_subgraphs = _ref_num_subgraphs;
    targetDevice = _targetDevice;
    init_input_shapes({inputShape});
    auto f = ov::test::snippets::AddRollConstFunction({inputDynamicShapes}, constShape);
    function = f.getOriginal();
    setInferenceType(type);
    setIgnoreCallbackMode();

    if (type == ov::element::bf16) {
        abs_threshold = 3e-2;
    }
}

std::string AddPair::getTestCaseName(const testing::TestParamInfo<ov::test::snippets::AddParamsPair>& obj) {
    const auto& [input_shapes, type, num_nodes, num_subgraphs, targetDevice] = obj.param;
    OPENVINO_ASSERT(input_shapes.size() == 2, "Invalid input shapes vector size");
    std::ostringstream result;
    result << "IS[0]=" << ov::test::utils::partialShape2str({input_shapes[0].first}) << "_";
    result << "TS[0]=";
    for (const auto& shape : input_shapes[0].second) {
        result << "(" << ov::test::utils::vec2str(shape) << ")_";
    }
    result << "IS[1]=" << ov::test::utils::partialShape2str({input_shapes[1].first}) << "_";
    result << "TS[1]=";
    for (const auto& shape : input_shapes[1].second) {
        result << "(" << ov::test::utils::vec2str(shape) << ")_";
    }
    result << "T=" << type << "_";
    result << "#N=" << num_nodes << "_";
    result << "#S=" << num_subgraphs << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void AddPair::SetUp() {
    const auto& [input_shapes, type, _ref_num_nodes, _ref_num_subgraphs, _targetDevice] = this->GetParam();
    ref_num_nodes = _ref_num_nodes;
    ref_num_subgraphs = _ref_num_subgraphs;
    targetDevice = _targetDevice;
    init_input_shapes(input_shapes);
    auto f = ov::test::snippets::AddFunction(inputDynamicShapes);
    function = f.getOriginal();
    setInferenceType(type);
    setIgnoreCallbackMode();
}

TEST_P(Add, CompareWithRefImpl) {
    run();
    validateNumSubgraphs();
}

TEST_P(AddConst, CompareWithRefImpl) {
    run();
    validateNumSubgraphs();
}

TEST_P(AddRollConst, CompareWithRefImpl) {
    run();
    validateNumSubgraphs();
}

TEST_P(AddPair, CompareWithRefImpl) {
    run();
    validateNumSubgraphs();
}

} // namespace snippets
} // namespace test
} // namespace ov
