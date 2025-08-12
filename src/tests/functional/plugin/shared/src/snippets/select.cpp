// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/common_utils.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>
#include "snippets/select.hpp"
#include "subgraph_simple.hpp"

namespace ov {
namespace test {
namespace snippets {

namespace {
void generate_data(std::map<std::shared_ptr<ov::Node>, ov::Tensor>& data_inputs, const std::vector<ov::Output<ov::Node>>& model_inputs,
    const std::vector<ov::Shape>& targetInputStaticShapes) {
    data_inputs.clear();
    ov::test::utils::InputGenerateData in_data;
    in_data.start_from = -1;
    in_data.range = 3;
    in_data.resolution = 2;
    auto tensor_bool = ov::test::utils::create_and_fill_tensor(model_inputs[0].get_element_type(), targetInputStaticShapes[0], in_data);

    in_data.start_from = -10;
    in_data.range = 10;
    in_data.resolution = 2;
    auto tensor0 = ov::test::utils::create_and_fill_tensor(model_inputs[1].get_element_type(), targetInputStaticShapes[1], in_data);

    in_data.start_from = 0;
    in_data.range = 10;
    in_data.resolution = 2;
    auto tensor1 = ov::test::utils::create_and_fill_tensor(model_inputs[2].get_element_type(), targetInputStaticShapes[2], in_data);
    data_inputs.insert({model_inputs[0].get_node_shared_ptr(), tensor_bool});
    data_inputs.insert({model_inputs[1].get_node_shared_ptr(), tensor0});
    data_inputs.insert({model_inputs[2].get_node_shared_ptr(), tensor1});
}
} // namespace

std::string Select::getTestCaseName(testing::TestParamInfo<ov::test::snippets::SelectParams> obj) {
    const auto& [inputShapes0, inputShapes1, inputShapes2, type, num_nodes, num_subgraphs, targetDevice] = obj.param;

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
    result << "IS[2]=" << ov::test::utils::partialShape2str({inputShapes2.first}) << "_";
    result << "TS[2]=";
    for (const auto& shape : inputShapes2.second) {
        result << "(" << ov::test::utils::vec2str(shape) << ")_";
    }
    result << "T=" << type << "_";
    result << "#N=" << num_nodes << "_";
    result << "#S=" << num_subgraphs << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void Select::SetUp() {
    const auto& [inputShape0, inputShape1, inputShape2, type, _ref_num_nodes, _ref_num_subgraphs, _targetDevice] =
        this->GetParam();
    ref_num_nodes = _ref_num_nodes;
    ref_num_subgraphs = _ref_num_subgraphs;
    targetDevice = _targetDevice;
    init_input_shapes({{inputShape0}, {inputShape1}, {inputShape2}});

    auto f = ov::test::snippets::SelectFunction(inputDynamicShapes);
    function = f.getOriginal();

    setIgnoreCallbackMode();
}

void Select::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    generate_data(inputs, function->inputs(), targetInputStaticShapes);
}

std::string BroadcastSelect::getTestCaseName(testing::TestParamInfo<ov::test::snippets::BroadcastSelectParams> obj) {
    const auto& [inputShapes0,
                 inputShapes1,
                 inputShapes2,
                 broadcastShape,
                 type,
                 num_nodes,
                 num_subgraphs,
                 targetDevice] = obj.param;

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
    result << "IS[2]=" << ov::test::utils::partialShape2str({inputShapes2.first}) << "_";
    result << "TS[2]=";
    for (const auto& shape : inputShapes2.second) {
        result << "(" << ov::test::utils::vec2str(shape) << ")_";
    }
    result << "IS_Broadcast=" << ov::test::utils::partialShape2str({broadcastShape}) << "_";
    result << "T=" << type << "_";
    result << "#N=" << num_nodes << "_";
    result << "#S=" << num_subgraphs << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void BroadcastSelect::SetUp() {
    const auto& [inputShape0,
                 inputShape1,
                 inputShape2,
                 broadcastShape,
                 type,
                 _ref_num_nodes,
                 _ref_num_subgraphs,
                 _targetDevice] = this->GetParam();
    ref_num_nodes = _ref_num_nodes;
    ref_num_subgraphs = _ref_num_subgraphs;
    targetDevice = _targetDevice;
    init_input_shapes({inputShape0, inputShape1, inputShape2});

    auto f = ov::test::snippets::BroadcastSelectFunction({inputDynamicShapes[0], inputDynamicShapes[1], inputDynamicShapes[2]}, broadcastShape);
    function = f.getOriginal();

    setIgnoreCallbackMode();
}

void BroadcastSelect::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    generate_data(inputs, function->inputs(), targetInputStaticShapes);
}

TEST_P(Select, CompareWithRefImpl) {
    run();
    validateNumSubgraphs();
}

TEST_P(BroadcastSelect, CompareWithRefImpl) {
    run();
    validateNumSubgraphs();
}

} // namespace snippets
} // namespace test
} // namespace ov
