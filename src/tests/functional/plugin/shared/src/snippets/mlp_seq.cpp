// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/mlp_seq.hpp"

#include <common_test_utils/ov_tensor_utils.hpp>

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "subgraph_mlp_seq.hpp"

namespace ov {
namespace test {
namespace snippets {

void MLPSeqBase::compile_model() {
    if (m_thread_count != default_thread_count)
        core->set_property(targetDevice, ov::inference_num_threads(m_thread_count));
    SubgraphBaseTest::compile_model();
}

void MLPSeqBase::SetUp() {
    std::vector<InputShape> input_shapes;
    ov::element::Type prc;
    ov::AnyMap additional_config;
    init_params(input_shapes, prc, additional_config);
    init_input_shapes(input_shapes);

    const auto subgraph_model = get_subgraph(m_num_hidden_layers, m_hidden_matmul_size);
    function = subgraph_model->getOriginal();

    configuration.insert(additional_config.begin(), additional_config.end());
    setIgnoreCallbackMode();

    inType = outType = prc;
    setInferenceType(prc);
}

std::string MLPSeq::getTestCaseName(testing::TestParamInfo<ov::test::snippets::MLPSeqParams> obj) {
    auto [input_shapes,
          elem_types,
          prc,
          thread_count,
          target_device,
          additional_config,
          num_hidden_layers_with_expectations,
          hidden_matmul_size] = obj.param;

    auto [num_hidden_layers, num_subgraphs_and_nodes] = num_hidden_layers_with_expectations;
    auto [num_subgraphs, num_nodes] = num_subgraphs_and_nodes;

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

void MLPSeq::init_params(std::vector<InputShape>& input_shapes, ov::element::Type& prc, ov::AnyMap& additional_config) {
    const auto& [_input_shapes,
                 _m_input_types,
                 _prc,
                 _m_thread_count,
                 _targetDevice,
                 _additional_config,
                 num_hidden_layers_with_expectations,
                 _m_hidden_matmul_size] = this->GetParam();
    input_shapes = _input_shapes;
    m_input_types = _m_input_types;
    prc = _prc;
    m_thread_count = _m_thread_count;
    targetDevice = _targetDevice;
    additional_config = _additional_config;
    m_hidden_matmul_size = _m_hidden_matmul_size;

    const auto& [_m_num_hidden_layers, ref_num_subgraphs_and_nodes] = num_hidden_layers_with_expectations;
    m_num_hidden_layers = _m_num_hidden_layers;
    std::tie(ref_num_subgraphs, ref_num_nodes) = ref_num_subgraphs_and_nodes;
}

std::shared_ptr<SnippetsFunctionBase> MLPSeq::get_subgraph(size_t num_hidden_layers, size_t hidden_matmul_size) const {
    return std::make_shared<ov::test::snippets::MLPSeqFunction>(inputDynamicShapes,
                                                                m_input_types,
                                                                num_hidden_layers,
                                                                hidden_matmul_size);
}

std::shared_ptr<SnippetsFunctionBase> MLPSeqQuantized::get_subgraph(size_t num_hidden_layers,
                                                                 size_t hidden_matmul_size) const {
    return std::make_shared<ov::test::snippets::MLPSeqQuantizedFunction>(inputDynamicShapes,
                                                                         m_input_types,
                                                                         num_hidden_layers,
                                                                         hidden_matmul_size);
}

TEST_P(MLPSeq, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
    validateNumSubgraphs();
}

TEST_P(MLPSeqQuantized, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
    validateNumSubgraphs();
}

}  // namespace snippets
}  // namespace test
}  // namespace ov
