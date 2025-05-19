// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/mlp.hpp"

#include <common_test_utils/ov_tensor_utils.hpp>

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "subgraph_mlp.hpp"

namespace ov {
namespace test {
namespace snippets {

void MLPBase::compile_model() {
    if (m_thread_count != default_thread_count)
        core->set_property(targetDevice, ov::inference_num_threads(m_thread_count));
    SubgraphBaseTest::compile_model();
}

void MLPBase::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
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

void MLPBase::SetUp() {
    std::vector<InputShape> input_shapes;
    ov::element::Type prc;
    ov::AnyMap additional_config;
    init_params(input_shapes, prc, additional_config);
    init_input_shapes(input_shapes);

    const auto subgraph_model = get_subgraph(m_num_hidden_layers, m_hidden_matmul_size);
    function = subgraph_model->getOriginal();

    configuration.insert(additional_config.begin(), additional_config.end());
    if (!configuration.count("SNIPPETS_MODE")) {
        configuration.insert({"SNIPPETS_MODE", "IGNORE_CALLBACK"});
    }

    inType = outType = prc;
    setInferenceType(prc);
    init_thresholds();
}

std::string MLP::getTestCaseName(testing::TestParamInfo<ov::test::snippets::MLPParams> obj) {
    auto [input_shapes,
          elem_types,
          prc,
          thread_count,
          num_nodes,
          num_subgraphs,
          target_device,
          additional_config,
          num_hidden_layers,
          hidden_matmul_size] = obj.param;

    std::ostringstream result;
    for (size_t i = 0; i < input_shapes.size(); i++)
        result << "IS[" << i << "]=" << input_shapes[i] << "_";
    for (size_t i = 0; i < elem_types.size(); i++)
        result << "T[" << i << "]=" << elem_types[i] << "_";
    result << "ThreadNum=" << thread_count << "_";
    result << "PRC=" << prc << "_";
    result << "#N=" << num_nodes << "_";
    result << "#S=" << num_subgraphs << "_";
    result << "targetDevice=" << target_device << "_";

    if (!additional_config.empty()) {
        result << "_PluginConf";
        for (auto& item : additional_config) {
            result << "_" << item.first << "=" << item.second.as<std::string>();
        }
    }
    result << "#num_hidden_layers=" << num_hidden_layers << "_";
    result << "#hidden_matmul_size=" << hidden_matmul_size << "_";
    return result.str();
}

void MLP::init_params(std::vector<InputShape>& input_shapes, ov::element::Type& prc, ov::AnyMap& additional_config) {
    std::tie(input_shapes,
             m_input_types,
             prc,
             m_thread_count,
             ref_num_nodes,
             ref_num_subgraphs,
             targetDevice,
             additional_config,
             m_num_hidden_layers,
             m_hidden_matmul_size) = this->GetParam();
}

std::shared_ptr<SnippetsFunctionBase> MLP::get_subgraph(size_t num_hidden_layers, size_t hidden_matmul_size) const {
    return std::make_shared<ov::test::snippets::MLPSeqFunction>(inputDynamicShapes,
                                                                m_input_types,
                                                                num_hidden_layers,
                                                                hidden_matmul_size);
}

std::shared_ptr<SnippetsFunctionBase> MLPQuantized::get_subgraph(size_t num_hidden_layers,
                                                                 size_t hidden_matmul_size) const {
    return std::make_shared<ov::test::snippets::MLPSeqQuantizedFunction>(inputDynamicShapes,
                                                                         m_input_types,
                                                                         num_hidden_layers,
                                                                         hidden_matmul_size);
}

TEST_P(MLP, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
    validateNumSubgraphs();
}

TEST_P(MLPQuantized, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
    validateNumSubgraphs();
}

}  // namespace snippets
}  // namespace test
}  // namespace ov
