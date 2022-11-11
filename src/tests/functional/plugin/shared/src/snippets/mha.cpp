// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/common_utils.hpp"
#include "snippets/mha.hpp"
#include "subgraph_mha.hpp"
#include "ngraph_functions/builders.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "cpp_interfaces/interface/ie_internal_plugin_config.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>

namespace ov {
namespace test {
namespace snippets {

std::string MHA::getTestCaseName(testing::TestParamInfo<ov::test::snippets::MHAParams> obj) {
    std::vector<ov::Shape> inputShapes;
    std::string targetDevice;
    size_t num_nodes, num_subgraphs;
    std::tie(inputShapes, num_nodes, num_subgraphs, targetDevice) = obj.param;

    std::ostringstream result;
    for (size_t i = 0; i < inputShapes.size(); ++i)
        result << "IS[" << i << "]=" << CommonTestUtils::vec2str(inputShapes[i]) << "_";
    result << "#N=" << num_nodes << "_";
    result << "#S=" << num_subgraphs << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void MHA::SetUp() {
    std::vector<ov::Shape> inputShapes;
    std::tie(inputShapes, ref_num_nodes, ref_num_subgraphs, targetDevice) = this->GetParam();
    init_input_shapes(static_shapes_to_test_representation(inputShapes));

    auto f = ov::test::snippets::MHASinhFunction(inputDynamicShapes);
    function = f.getOriginal();

    if (!configuration.count(InferenceEngine::PluginConfigInternalParams::KEY_SNIPPETS_MODE)) {
        configuration.insert({InferenceEngine::PluginConfigInternalParams::KEY_SNIPPETS_MODE,
                              InferenceEngine::PluginConfigInternalParams::IGNORE_CALLBACK});
    }
}

void MHASelect::SetUp() {
    std::vector<ov::Shape> inputShapes;
    std::tie(inputShapes, ref_num_nodes, ref_num_subgraphs, targetDevice) = this->GetParam();
    init_input_shapes(static_shapes_to_test_representation(inputShapes));

    auto f = ov::test::snippets::MHASelectSinhFunction(inputDynamicShapes);
    function = f.getOriginal();

    if (!configuration.count(InferenceEngine::PluginConfigInternalParams::KEY_SNIPPETS_MODE)) {
        configuration.insert({InferenceEngine::PluginConfigInternalParams::KEY_SNIPPETS_MODE,
                              InferenceEngine::PluginConfigInternalParams::IGNORE_CALLBACK});
    }
}

void MHASelect::generate_inputs(const std::vector<ngraph::Shape>& targetInputStaticShapes) {
    inputs.clear();
    auto model_inputs = function->inputs();
    for (auto& model_input : model_inputs) {
        const auto node_input = model_input.get_node_shared_ptr();
        const auto name = node_input->get_friendly_name();
        ov::Tensor tensor;
        int seed = 0;
        if (name.find("less") != std::string::npos) {
            tensor = ov::test::utils::create_and_fill_tensor(model_input.get_element_type(), model_input.get_shape(), 5 + seed, -2, 10, seed++);
        } else {
            tensor = ov::test::utils::create_and_fill_tensor(model_input.get_element_type(), model_input.get_shape());
        }
        inputs.insert({node_input, tensor});
    }
}


TEST_P(MHA, CompareWithRefImpl) {
    run();
    validateNumSubgraphs();
}

TEST_P(MHASelect, CompareWithRefImpl) {
    run();
    validateNumSubgraphs();
}

} // namespace snippets
} // namespace test
} // namespace ov
