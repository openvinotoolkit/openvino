// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/mha.hpp"

#include <common_test_utils/ov_tensor_utils.hpp>

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "subgraph_mha.hpp"

namespace ov {
namespace test {
namespace snippets {

void MHABase::compile_model() {
    if (m_thread_count != default_thread_count)
        core->set_property(targetDevice, ov::inference_num_threads(m_thread_count));
    SubgraphBaseTest::compile_model();
}

void MHABase::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
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

void MHABase::SetUp() {
    std::vector<InputShape> input_shapes;
    ov::element::Type prc;
    ov::AnyMap additional_config;
    init_params(input_shapes, prc, additional_config);
    init_input_shapes(input_shapes);

    const auto subgraph_model = get_subgraph();
    function = subgraph_model->getOriginal();

    configuration.insert(additional_config.begin(), additional_config.end());
    if (!configuration.count("SNIPPETS_MODE")) {
        configuration.insert({"SNIPPETS_MODE", "IGNORE_CALLBACK"});
    }

    inType = outType = prc;
    setInferenceType(prc);
    init_thresholds();
}

 void MHABase::init_thresholds() {
    // Note: Libxsmm calculates Exp in a slightly different way, so the abs values might differ a bit. Ticket: 130699
#ifdef SNIPPETS_LIBXSMM_TPP
    abs_threshold = 1e-6;
#endif
    if (inType == ov::element::bf16)
        rel_threshold = 0.05f;
    if (inType == ov::element::f16)
        abs_threshold = 2e-2;
 }

std::string MHA::getTestCaseName(testing::TestParamInfo<ov::test::snippets::MHAParams> obj) {
    std::vector<InputShape> input_shapes;
    std::vector<ov::element::Type> elem_types;
    ov::element::Type prc;
    bool with_mul;
    size_t thread_count;
    std::string target_device;
    size_t num_nodes, num_subgraphs;
    ov::AnyMap additional_config;
    std::tie(input_shapes,
             elem_types,
             prc,
             with_mul,
             thread_count,
             num_nodes,
             num_subgraphs,
             target_device,
             additional_config) = obj.param;

    std::ostringstream result;
    for (size_t i = 0; i < input_shapes.size(); i++)
        result << "IS[" << i << "]=" << input_shapes[i] << "_";
    for (size_t i = 0; i < elem_types.size(); i++)
        result << "T[" << i << "]=" << elem_types[i] << "_";
    result << "Mul=" << with_mul << "_";
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
    return result.str();
}

std::string MHAWithDynamicMul::getTestCaseName(testing::TestParamInfo<ov::test::snippets::MHAWithDynamicMulParams> obj) {
    std::vector<InputShape> input_shapes;
    std::vector<ov::element::Type> elem_types;
    ov::element::Type prc;
    size_t thread_count;
    std::string target_device;
    size_t num_nodes, num_subgraphs;
    ov::AnyMap additional_config;
    std::tie(input_shapes, elem_types, prc, thread_count, num_nodes, num_subgraphs, target_device, additional_config) = obj.param;

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
    return result.str();
}

void MHA::init_params(std::vector<InputShape>& input_shapes, ov::element::Type& prc, ov::AnyMap& additional_config) {
    std::tie(input_shapes, m_input_types, prc, m_with_mul, m_thread_count, ref_num_nodes, ref_num_subgraphs,
             targetDevice, additional_config) = this->GetParam();
}

void MHAWithDynamicMul::init_params(std::vector<InputShape>& input_shapes, ov::element::Type& prc, ov::AnyMap& additional_config) {
    std::tie(input_shapes, m_input_types, prc, m_thread_count, ref_num_nodes, ref_num_subgraphs, targetDevice, additional_config) = this->GetParam();
}

std::shared_ptr<SnippetsFunctionBase> MHA::get_subgraph() const {
    bool is_with_reshape = std::all_of(inputDynamicShapes.begin(), inputDynamicShapes.end(), [](const PartialShape& ps){ return ps.is_static(); });
    return std::make_shared<ov::test::snippets::MHAFunction>(inputDynamicShapes, m_input_types, m_with_mul, is_with_reshape);
}

void MHA::init_thresholds() {
    MHABase::init_thresholds();
    auto precision_hint = configuration.count(ov::hint::inference_precision.name())
                              ? configuration.at(ov::hint::inference_precision.name())
                              : ov::element::f32;
    if (m_input_types.size() > 1 && m_input_types[0] == ov::element::bf16 && precision_hint == ov::element::f32) {
        rel_threshold = 0.01f;
        abs_threshold = 0.0078125f;
    }
}

void MHASelect::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    inputs.clear();
    auto model_inputs = function->inputs();
    for (auto& model_input : model_inputs) {
        const auto node_input = model_input.get_node_shared_ptr();
        const auto name = node_input->get_friendly_name();
        ov::Tensor tensor;
        int seed = 0;
        if (name.find("less") != std::string::npos) {
            ov::test::utils::InputGenerateData in_data;
            in_data.start_from = -2;
            in_data.range = 5 + seed;
            in_data.resolution = 10;
            in_data.seed = seed;
            tensor = ov::test::utils::create_and_fill_tensor(model_input.get_element_type(),
                                                             model_input.get_shape(),
                                                             in_data);
            seed++;
        } else {
            ov::test::utils::InputGenerateData in_data;
            in_data.start_from = -1;
            in_data.range = 2;
            in_data.resolution = 256;
            tensor = ov::test::utils::create_and_fill_tensor(model_input.get_element_type(),
                                                             model_input.get_shape(),
                                                             in_data);
        }
        inputs.insert({node_input, tensor});
    }
}

std::shared_ptr<SnippetsFunctionBase> MHASelect::get_subgraph() const {
    return std::make_shared<ov::test::snippets::MHASelectFunction>(inputDynamicShapes, m_input_types);
}

std::shared_ptr<SnippetsFunctionBase> MHAWOTransposeOnInputs::get_subgraph() const {
    return std::make_shared<ov::test::snippets::MHAWOTransposeOnInputsFunction>(inputDynamicShapes);
}

std::shared_ptr<SnippetsFunctionBase> MHAWOTranspose::get_subgraph() const {
    return std::make_shared<ov::test::snippets::MHAWOTransposeFunction>(inputDynamicShapes, m_input_types);
}

std::shared_ptr<SnippetsFunctionBase> MHAINT8MatMul::get_subgraph() const {
    return std::make_shared<ov::test::snippets::MHAINT8MatMulFunction>(inputDynamicShapes);
}

void MHAINT8MatMul::init_thresholds() {
    MHABase::init_thresholds();
    abs_threshold = 4e-6;
}

std::shared_ptr<SnippetsFunctionBase> MHAQuantMatMul0::get_subgraph() const {
    return std::make_shared<ov::test::snippets::MHAQuantMatMul0Function>(inputDynamicShapes);
}

std::shared_ptr<SnippetsFunctionBase> MHAFQAfterMatMul::get_subgraph() const {
    return std::make_shared<ov::test::snippets::MHAFQAfterMatMulFunction>(inputDynamicShapes);
}

std::shared_ptr<SnippetsFunctionBase> MHAFQ::get_subgraph() const {
    return std::make_shared<ov::test::snippets::MHAFQFunction>(inputDynamicShapes);
}

void MHAFQ::init_thresholds() {
    MHABase::init_thresholds();
    abs_threshold = 0.016;
}

std::shared_ptr<SnippetsFunctionBase> MHAMulAdd::get_subgraph() const {
    return std::make_shared<ov::test::snippets::MHAMulAddFunction>(inputDynamicShapes);
}

std::shared_ptr<SnippetsFunctionBase> MHATransposedB::get_subgraph() const {
    return std::make_shared<ov::test::snippets::MHATransposedInputFunction>(inputDynamicShapes, true);
}

std::shared_ptr<SnippetsFunctionBase> MHAWithExtractedReshape::get_subgraph() const {
    return std::make_shared<ov::test::snippets::MHAWithExtractedReshapeFunction>(inputDynamicShapes, false);
}

std::shared_ptr<SnippetsFunctionBase> MHAWithDynamicMul::get_subgraph() const {
    return std::make_shared<ov::test::snippets::MHAWithDynamicMulFunction>(inputDynamicShapes, m_input_types);
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
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
    validateNumSubgraphs();
}

TEST_P(MHAMulAdd, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
    validateNumSubgraphs();
}

TEST_P(MHATransposedB, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
    validateNumSubgraphs();
}

TEST_P(MHAINT8MatMul, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
    validateNumSubgraphs();
}

TEST_P(MHAQuantMatMul0, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
    validateNumSubgraphs();
}

TEST_P(MHAFQAfterMatMul, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    abs_threshold = 4e-6;
    run();
    validateNumSubgraphs();
}

TEST_P(MHAFQ, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
    validateNumSubgraphs();
}

TEST_P(MHAWithExtractedReshape, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
    validateNumSubgraphs();
}

TEST_P(MHAWithDynamicMul, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
    validateNumSubgraphs();
}

}  // namespace snippets
}  // namespace test
}  // namespace ov
