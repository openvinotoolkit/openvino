// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/common_utils.hpp"
#include "snippets/mha.hpp"
#include "subgraph_mha.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "cpp_interfaces/interface/ie_internal_plugin_config.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>

namespace ov {
namespace test {
namespace snippets {

std::string MHA::getTestCaseName(testing::TestParamInfo<ov::test::snippets::MHAParams> obj) {
    std::vector<ov::PartialShape> inputShapes;
    bool withMul;
    ov::element::Type prc;
    std::string targetDevice;
    size_t num_nodes, num_subgraphs;
    std::map<std::string, std::string> additionalConfig;
    std::tie(inputShapes, withMul, prc, num_nodes, num_subgraphs, targetDevice, additionalConfig) = obj.param;

    std::ostringstream result;
    for (size_t i = 0; i < inputShapes.size(); ++i)
        result << "IS[" << i << "]=" << CommonTestUtils::partialShape2str({inputShapes[i]}) << "_";
    result << "Mul=" << withMul << "_";
    result << "PRC=" << prc << "_";
    result << "#N=" << num_nodes << "_";
    result << "#S=" << num_subgraphs << "_";
    result << "targetDevice=" << targetDevice << "_";

    if (!additionalConfig.empty()) {
        result << "_PluginConf";
        for (auto &item : additionalConfig) {
            if (item.second == InferenceEngine::PluginConfigParams::YES)
                result << "_" << item.first << "=" << item.second;
        }
    }
    return result.str();
}

void MHA::SetUp() {
    std::vector<ov::PartialShape> inputShapes;
    ov::element::Type prc;
    std::map<std::string, std::string> additionalConfig;
    std::tie(inputShapes, m_with_mul, prc, ref_num_nodes, ref_num_subgraphs, targetDevice, additionalConfig) = this->GetParam();
    init_input_shapes(static_partial_shapes_to_test_representation(inputShapes));

    init_subgraph();

    configuration.insert(additionalConfig.begin(), additionalConfig.end());
    if (additionalConfig.empty() && !configuration.count(InferenceEngine::PluginConfigInternalParams::KEY_SNIPPETS_MODE)) {
        configuration.insert({InferenceEngine::PluginConfigInternalParams::KEY_SNIPPETS_MODE,
                              InferenceEngine::PluginConfigInternalParams::IGNORE_CALLBACK});
    }

    setInferenceType(prc);
    inType = outType = prc;
    if (prc == ov::element::bf16)
        abs_threshold = 0.3;
}

void MHA::generate_inputs(const std::vector<ngraph::Shape>& targetInputStaticShapes) {
    inputs.clear();
    const auto& model_inputs = function->inputs();
    for (int i = 0; i < model_inputs.size(); ++i) {
        const auto& model_input = model_inputs[i];
        ov::Tensor tensor;
        tensor = ov::test::utils::create_and_fill_tensor_normal_distribution(model_input.get_element_type(), targetInputStaticShapes[i], 1.0f, 0.5f);
        inputs.insert({model_input.get_node_shared_ptr(), tensor});
    }
}

void MHA::init_subgraph() {
    auto f = ov::test::snippets::MHAFunction(inputDynamicShapes, m_with_mul);
    function = f.getOriginal();
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
            tensor = ov::test::utils::create_and_fill_tensor(model_input.get_element_type(), model_input.get_shape(), 5 + seed, -2, 10, seed);
            seed++;
        } else {
            tensor = ov::test::utils::create_and_fill_tensor_normal_distribution(model_input.get_element_type(), model_input.get_shape(), 1.0f, 0.5f);
        }
        inputs.insert({node_input, tensor});
    }
}

void MHASelect::init_subgraph() {
    auto f = ov::test::snippets::MHASelectFunction(inputDynamicShapes);
    function = f.getOriginal();
}

void MHAWOTransposeOnInputs::init_subgraph() {
    auto f = ov::test::snippets::MHAWOTransposeOnInputsFunction(inputDynamicShapes);
    function = f.getOriginal();
}

void MHAWOTranspose::init_subgraph() {
    auto f = ov::test::snippets::MHAWOTransposeFunction(inputDynamicShapes);
    function = f.getOriginal();
}

TEST_P(MHA, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
    validateNumSubgraphs();
}

TEST_P(MHASelect, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
    validateNumSubgraphs();
}

TEST_P(MHAWOTransposeOnInputs, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
    validateNumSubgraphs();
}

TEST_P(MHAWOTranspose, CompareWithRefImpl) {
    run();
    validateNumSubgraphs();
}


} // namespace snippets
} // namespace test
} // namespace ov
