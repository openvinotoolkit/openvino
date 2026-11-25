// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "grouped_matmul.hpp"

#include <memory>
#include <sstream>
#include <string>

#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/subgraph_builders/weights_decompression_builders.hpp"
#include "gtest/gtest.h"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/runtime/exec_model_info.hpp"
#include "openvino/runtime/properties.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

namespace {

constexpr auto groupedMatMulLayerType = "GroupedMatMul";

// The executed GroupedMatMul, found by layer type. Only used to inspect its input precisions -
// presence is asserted via CheckNumberOfNodesWithType.
std::shared_ptr<ov::Node> executed_grouped_matmul(const ov::CompiledModel& compiled_model) {
    for (const auto& node : compiled_model.get_runtime_model()->get_ops()) {
        const auto& rt_info = node->get_rt_info();
        const auto it = rt_info.find(ov::exec_model_info::LAYER_TYPE);
        if (it != rt_info.end() && it->second.as<std::string>() == groupedMatMulLayerType) {
            return node;
        }
    }
    return nullptr;
}

std::string shapes_to_string(const GroupedMatMulShapeParams& shape_params) {
    std::ostringstream result;
    result << "A_shape=" << shape_params.a_input_shape << "_";
    result << "B_shape=" << ov::test::utils::vec2str(shape_params.b_shape) << "_";
    return result.str();
}

std::string config_to_string(const ov::AnyMap& config) {
    std::ostringstream result;
    result << "config=(";
    for (const auto& [key, value] : config) {
        result << key << "=" << value.as<std::string>() << "_";
    }
    result << ")";
    return result.str();
}

// bf16 is requested through the property, not through the model precision: the compression pass runs
// before Graph::EnforceInferencePrecision, so it has to see f32 activations at conversion time.
ov::element::Type configured_inference_precision(const ov::AnyMap& config) {
    const auto it = config.find(ov::hint::inference_precision.name());
    return it == config.end() ? ov::element::dynamic : it->second.as<ov::element::Type>();
}

}  // namespace

// ---- GroupedMatMulLayerCPUTest ----------------------------------------------------------------

std::string GroupedMatMulLayerCPUTest::get_test_case_name(const testing::TestParamInfo<GroupedMatMulCPUTestParams>& obj) {
    const auto& [shape_params, act_type, config, cpu_params] = obj.param;
    std::ostringstream result;
    result << shapes_to_string(shape_params);
    result << "ActET=" << act_type << "_";
    result << config_to_string(config);
    result << CPUTestsBase::getTestCaseName(cpu_params);
    return result.str();
}

void GroupedMatMulLayerCPUTest::SetUp() {
    const auto& [shape_params, act_type, config, cpu_params] = GetParam();
    shape_params_ = shape_params;
    act_type_ = act_type;
    model_name_ = "GroupedMatMul";
    // CheckPluginRelatedResults inspects the executable graph, so the base profiling check is off
    expected_primitive_.clear();
    targetDevice = ov::test::utils::DEVICE_CPU;
    configuration.insert(config.begin(), config.end());

    std::tie(inFmts, outFmts, priority, selectedType) = cpu_params;
    if (selectedType.empty()) {
        selectedType = getPrimitiveType();
    }
    selectedType = makeSelectedTypeStr(selectedType, act_type);

    GroupedMatMulTestBase::SetUp();

    // The model stays f32; bf16 comes from Graph::EnforceInferencePrecision, so the reference is
    // computed in a higher precision than the execution
    if (configured_inference_precision(configuration) == ov::element::bf16) {
        abs_threshold = 0.2F;
        rel_threshold = 0.1F;
    }
}

void GroupedMatMulLayerCPUTest::validate() {
    GroupedMatMulTestBase::validate();
    // CheckPluginRelatedResults only validates the nodes it finds, so assert presence separately
    CheckNumberOfNodesWithType(compiledModel, groupedMatMulLayerType, 1);
    CheckPluginRelatedResults(compiledModel, groupedMatMulLayerType);
}

TEST_P(GroupedMatMulLayerCPUTest, CompareWithRefs) {
    run();
}

// Instantiated in instances/x64 only, which non-x64 builds exclude while still compiling this file.
// gtest >= 1.11 fails an uninstantiated suite unless it is explicitly allowed.
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(GroupedMatMulLayerCPUTest);

// ---- GroupedMatMulCompressedLayerCPUTest ------------------------------------------------------

std::string GroupedMatMulCompressedLayerCPUTest::get_test_case_name(
    const testing::TestParamInfo<GroupedMatMulCompressedCPUTestParams>& obj) {
    const auto& [shape_params,
                 act_type,
                 weights_prec,
                 decomp_prec,
                 scale_prec,
                 multiply_type,
                 subtract_type,
                 reshape_on_decomp,
                 group_size,
                 expect_compressed,
                 config,
                 cpu_params] = obj.param;
    std::ostringstream result;
    result << shapes_to_string(shape_params);
    result << "ActET=" << act_type << "_";
    result << "WET=" << weights_prec << "_";
    result << "DecompPrec=" << decomp_prec << "_";
    result << "ScalePrec=" << scale_prec << "_";
    result << "Mul=" << multiply_type << "_Sub=" << subtract_type << "_";
    result << "Reshape=" << reshape_on_decomp << "_";
    result << "GrpSz=" << group_size << "_";
    result << "Compressed=" << expect_compressed << "_";
    result << config_to_string(config);
    result << CPUTestsBase::getTestCaseName(cpu_params);
    return result.str();
}

void GroupedMatMulCompressedLayerCPUTest::SetUp() {
    const auto& [shape_params,
                 act_type,
                 weights_prec,
                 decomp_prec,
                 scale_prec,
                 multiply_type,
                 subtract_type,
                 reshape_on_decomp,
                 group_size,
                 expect_compressed,
                 config,
                 cpu_params] = GetParam();
    shape_params_ = shape_params;
    act_type_ = act_type;
    model_name_ = "GroupedMatMulCompressed";
    expected_primitive_.clear();
    m_weights_prec = weights_prec;
    m_decomp_prec = decomp_prec;
    m_scale_prec = scale_prec;
    m_multiply_type = multiply_type;
    m_subtract_type = subtract_type;
    m_reshape_on_decomp = reshape_on_decomp;
    m_group_size = group_size;
    m_expect_compressed = expect_compressed;
    targetDevice = ov::test::utils::DEVICE_CPU;
    configuration.insert(config.begin(), config.end());

    std::tie(inFmts, outFmts, priority, selectedType) = cpu_params;
    if (selectedType.empty()) {
        selectedType = getPrimitiveType();
    }
    selectedType = makeSelectedTypeStr(selectedType, act_type);

    GroupedMatMulTestBase::SetUp();

    set_compressed_thresholds();
    // Dynamic quantization of activations (s8) adds extra error
    if (configuration.count(ov::hint::dynamic_quantization_group_size.name()) != 0U) {
        abs_threshold = std::max(abs_threshold, 0.6);
        rel_threshold = 0.1F;
    }
    if (configured_inference_precision(configuration) == ov::element::bf16) {
        abs_threshold = 0.5F;
        rel_threshold = 0.1F;
    }
}

void GroupedMatMulCompressedLayerCPUTest::validate() {
    GroupedMatMulTestBase::validate();
    CheckNumberOfNodesWithType(compiledModel, groupedMatMulLayerType, 1);
    CheckPluginRelatedResults(compiledModel, groupedMatMulLayerType);

    const auto node = executed_grouped_matmul(compiledModel);
    ASSERT_NE(nullptr, node);
    // When the compressed primitive is used the weights stay in their low-precision form all the way
    // into the executor; otherwise the dequantization subgraph is folded and f32 weights arrive.
    const auto expected_weights_precision = m_expect_compressed ? m_weights_prec : node->get_input_element_type(0);
    EXPECT_EQ(node->get_input_element_type(1), expected_weights_precision);
}

TEST_P(GroupedMatMulCompressedLayerCPUTest, CompareWithRefs) {
    run();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(GroupedMatMulCompressedLayerCPUTest);

}  // namespace test
}  // namespace ov
