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

using namespace CPUTestUtils;

namespace ov {
namespace test {

namespace {

constexpr auto groupedMatMulLayerType = "GroupedMatMul";

// The executed GroupedMatMul, looked up in the executable graph by its layer type. Presence is
// asserted separately via CheckNumberOfNodesWithType; this only fetches the node so that its input
// precisions can be inspected.
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

}  // namespace

// ---- GroupedMatMulLayerCPUTest ----------------------------------------------------------------

std::string GroupedMatMulLayerCPUTest::getTestCaseName(const testing::TestParamInfo<GroupedMatMulCPUTestParams>& obj) {
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
}

std::shared_ptr<ov::Node> GroupedMatMulLayerCPUTest::build_weights() {
    ov::test::utils::InputGenerateData b_data;
    b_data.range = 2;
    b_data.resolution = 128;
    b_data.start_from = -1;
    auto b_tensor = ov::test::utils::create_and_fill_tensor(act_type_, shape_params_.b_shape, b_data);
    return std::make_shared<ov::op::v0::Constant>(b_tensor);
}

void GroupedMatMulLayerCPUTest::check_results() {
    // CheckPluginRelatedResults only validates the nodes it finds, so the presence of exactly one
    // GroupedMatMul is asserted separately.
    CheckNumberOfNodesWithType(compiledModel, groupedMatMulLayerType, 1);
    CheckPluginRelatedResults(compiledModel, groupedMatMulLayerType);
}

TEST_P(GroupedMatMulLayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
    check_results();
}

// ---- GroupedMatMulCompressedLayerCPUTest ------------------------------------------------------

std::string GroupedMatMulCompressedLayerCPUTest::getTestCaseName(
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
    weights_prec_ = weights_prec;
    decomp_prec_ = decomp_prec;
    scale_prec_ = scale_prec;
    multiply_type_ = multiply_type;
    subtract_type_ = subtract_type;
    reshape_on_decomp_ = reshape_on_decomp;
    group_size_ = group_size;
    expect_compressed_ = expect_compressed;
    targetDevice = ov::test::utils::DEVICE_CPU;
    configuration.insert(config.begin(), config.end());

    std::tie(inFmts, outFmts, priority, selectedType) = cpu_params;
    if (selectedType.empty()) {
        selectedType = getPrimitiveType();
    }
    selectedType = makeSelectedTypeStr(selectedType, act_type);

    GroupedMatMulTestBase::SetUp();

    // Dequantization adds rounding error on top of the uncompressed path
    if (weights_prec_ == ov::element::u4 || weights_prec_ == ov::element::i4) {
        abs_threshold = 0.2f;
    } else if (weights_prec_ == ov::element::u8 || weights_prec_ == ov::element::i8) {
        abs_threshold = 0.1f;
    }
}

std::shared_ptr<ov::Node> GroupedMatMulCompressedLayerCPUTest::build_weights() {
    // b_shape is the pre-transposed weight shape [G, N, K]. The planar shape [G, K, N] is handed to
    // the builder with transpose_weights=true so the constant ends up as [G, N, K] (per-N scales are
    // per-OC), and insert_transpose_node=false suppresses the trailing Transpose, leaving the
    // decompressed output already in the [G, N, K] layout GroupedMatMul expects.
    ov::Shape b_planar = shape_params_.b_shape;
    std::swap(b_planar[b_planar.size() - 2], b_planar[b_planar.size() - 1]);

    return ov::test::utils::initMatMulDecompressionSubgraphQuantization(b_planar,
                                                                        group_size_,
                                                                        act_type_,
                                                                        weights_prec_,
                                                                        decomp_prec_,
                                                                        scale_prec_,
                                                                        true,
                                                                        multiply_type_,
                                                                        subtract_type_,
                                                                        reshape_on_decomp_,
                                                                        false,
                                                                        1);
}

void GroupedMatMulCompressedLayerCPUTest::check_results() {
    CheckNumberOfNodesWithType(compiledModel, groupedMatMulLayerType, 1);
    CheckPluginRelatedResults(compiledModel, groupedMatMulLayerType);

    const auto node = executed_grouped_matmul(compiledModel);
    ASSERT_NE(nullptr, node);
    // When the compressed primitive is used the weights stay in their low-precision form all the way
    // into the executor; otherwise the dequantization subgraph is folded and f32 weights arrive.
    const auto expected_weights_precision = expect_compressed_ ? weights_prec_ : node->get_input_element_type(0);
    EXPECT_EQ(node->get_input_element_type(1), expected_weights_precision);
}

TEST_P(GroupedMatMulCompressedLayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
    check_results();
}

}  // namespace test
}  // namespace ov
