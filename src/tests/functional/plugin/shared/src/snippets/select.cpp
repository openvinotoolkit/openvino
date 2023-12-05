// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/common_utils.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>
#include "cpp_interfaces/interface/ie_internal_plugin_config.hpp"
#include "snippets/select.hpp"
#include "subgraph_simple.hpp"

namespace ov {
namespace test {
namespace snippets {

namespace {
void generate_data(std::map<std::shared_ptr<ov::Node>, ov::Tensor>& data_inputs, const std::vector<ov::Output<ov::Node>>& model_inputs,
    const std::vector<ngraph::Shape>& targetInputStaticShapes) {
    data_inputs.clear();
    auto tensor_bool = ov::test::utils::create_and_fill_tensor(model_inputs[0].get_element_type(), targetInputStaticShapes[0], 3, -1, 2);
    auto tensor0 = ov::test::utils::create_and_fill_tensor(model_inputs[1].get_element_type(), targetInputStaticShapes[1], 10, -10, 2);
    auto tensor1 = ov::test::utils::create_and_fill_tensor(model_inputs[2].get_element_type(), targetInputStaticShapes[2], 10, 0, 2);
    data_inputs.insert({model_inputs[0].get_node_shared_ptr(), tensor_bool});
    data_inputs.insert({model_inputs[1].get_node_shared_ptr(), tensor0});
    data_inputs.insert({model_inputs[2].get_node_shared_ptr(), tensor1});
}
} // namespace

std::string Select::getTestCaseName(testing::TestParamInfo<ov::test::snippets::SelectParams> obj) {
    InputShape inputShapes0, inputShapes1, inputShapes2;
    ov::element::Type type;
    std::string targetDevice;
    size_t num_nodes, num_subgraphs;
    std::tie(inputShapes0, inputShapes1, inputShapes2, type, num_nodes, num_subgraphs, targetDevice) = obj.param;

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
    InputShape inputShape0, inputShape1, inputShape2;
    ov::element::Type type;
    std::tie(inputShape0, inputShape1, inputShape2, type, ref_num_nodes, ref_num_subgraphs, targetDevice) = this->GetParam();
    init_input_shapes({{inputShape0}, {inputShape1}, {inputShape2}});

    auto f = ov::test::snippets::SelectFunction(inputDynamicShapes);
    function = f.getOriginal();

    if (!configuration.count(InferenceEngine::PluginConfigInternalParams::KEY_SNIPPETS_MODE)) {
        configuration.insert({InferenceEngine::PluginConfigInternalParams::KEY_SNIPPETS_MODE,
                              InferenceEngine::PluginConfigInternalParams::IGNORE_CALLBACK});
    }
}

void Select::generate_inputs(const std::vector<ngraph::Shape>& targetInputStaticShapes) {
    generate_data(inputs, function->inputs(), targetInputStaticShapes);
}

std::string BroadcastSelect::getTestCaseName(testing::TestParamInfo<ov::test::snippets::BroadcastSelectParams> obj) {
    InputShape inputShapes0, inputShapes1, inputShapes2;
    ov::PartialShape broadcastShape;
    ov::element::Type type;
    std::string targetDevice;
    size_t num_nodes, num_subgraphs;
    std::tie(inputShapes0, inputShapes1, inputShapes2, broadcastShape, type, num_nodes, num_subgraphs, targetDevice) = obj.param;

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
    InputShape inputShape0, inputShape1, inputShape2;
    ov::PartialShape broadcastShape;
    ov::element::Type type;
    std::tie(inputShape0, inputShape1, inputShape2, broadcastShape, type, ref_num_nodes, ref_num_subgraphs, targetDevice) = this->GetParam();
    init_input_shapes({inputShape0, inputShape1, inputShape2});

    auto f = ov::test::snippets::BroadcastSelectFunction({inputDynamicShapes[0], inputDynamicShapes[1], inputDynamicShapes[2]}, broadcastShape);
    function = f.getOriginal();

    if (!configuration.count(InferenceEngine::PluginConfigInternalParams::KEY_SNIPPETS_MODE)) {
        configuration.insert({InferenceEngine::PluginConfigInternalParams::KEY_SNIPPETS_MODE,
                              InferenceEngine::PluginConfigInternalParams::IGNORE_CALLBACK});
    }
}

void BroadcastSelect::generate_inputs(const std::vector<ngraph::Shape>& targetInputStaticShapes) {
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
