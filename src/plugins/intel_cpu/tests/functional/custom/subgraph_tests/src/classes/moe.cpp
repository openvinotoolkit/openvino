// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <sstream>
#include <vector>

#include "common_test_utils/node_builders/moe_builders.hpp"
#include "common_test_utils/subgraph_builders/weights_decompression_builders.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "shared_test_classes/subgraph/weights_decompression_params.hpp"
#include "utils/cpu_test_utils.hpp"
#include "utils/fusing_test_utils.hpp"
#include "moe.hpp"

using namespace CPUTestUtils;

namespace ov::test {

inline std::ostream& operator<<(std::ostream& os, const MoEType& type) {
    switch (type) {
    case MoEType::MoE2GeMM:
        return os << "MoE2GeMM";
    case MoEType::MoE3GeMM:
        return os << "MoE3GeMM";
    default:
        OPENVINO_THROW("Unsupported MoEType");
    }
}

inline std::ostream& operator<<(std::ostream& os, const MoEActivationType& type) {
    switch (type) {
    case MoEActivationType::SWISH:
        return os << "SWISH";
    case MoEActivationType::GELU:
        return os << "GELU";
    default:
        OPENVINO_THROW("Unsupported MoEActivationType");
    }
}

std::string MoESubgraphTest::generateBaseMoeTestName(const MoeTestShapeParams& moe_params,
                                               const MoEType& moe_type,
                                               const ov::AnyMap& additional_config) {
    std::ostringstream result;
    result << "IS=" << ov::test::utils::partialShape2str({moe_params.data_shape.first}) << "_";
    result << "TS=";
    for (const auto& static_shape : moe_params.data_shape.second) {
        result << ov::test::utils::vec2str(static_shape) << ",";
    }
    result << "top_k_experts=" << moe_params.topk << "_";
    result << "total_experts=" << moe_params.number_of_experts << "_";
    result << "intermediate_size=" << moe_params.intermediate_size << "_";
    result << "moe_type=" << moe_type << "_";

    result << "config=(";
    for (const auto& configEntry : additional_config) {
        result << configEntry.first << "=" << configEntry.second.as<std::string>() << "_";
    }
    result << ")";

    return result.str();
}

std::string MoESubgraphTest::getTestCaseName(const testing::TestParamInfo<MoeTestParams>& obj) {
    const auto& [moe_params, moe_type, activation_type, additional_config] = obj.param;
    std::ostringstream result;
    result << generateBaseMoeTestName(moe_params, moe_type, additional_config);
    result << "_act=" << activation_type;
    return result.str();
}

size_t MoESubgraphTest::get_expected_gather_mm_count(MoEType moe_type) {
    switch (moe_type) {
    case MoEType::MoE2GeMM:
        return 2;
    case MoEType::MoE3GeMM:
        return 3;
    default:
        OPENVINO_THROW("Unsupported MoEType");
    }
}

std::set<std::shared_ptr<ov::Node>> MoESubgraphTest::get_gather_mm_nodes(const std::shared_ptr<const ov::Model>& model,
                                                                        MoEType moe_type) {
    const std::string expected_gather_mm_type = "GatherMatmul";
    std::set<std::shared_ptr<ov::Node>> gather_mm_nodes;
    for (const auto& node : model->get_ordered_ops()) {
        if (node->get_rt_info().at(ov::exec_model_info::LAYER_TYPE).as<std::string>() == expected_gather_mm_type) {
            gather_mm_nodes.insert(node);
        }
    }
    return gather_mm_nodes;
}

void MoESubgraphTest::SetUp() {
    targetDevice = ov::test::utils::DEVICE_CPU;
    const auto& [moe_params, moe_type, activation_type, additional_config] = GetParam();

    configuration.insert(additional_config.begin(), additional_config.end());
    init_input_shapes({moe_params.data_shape});
    const MoePatternParams shape_params{moe_params.data_shape.first,
                                        moe_params.topk,
                                        moe_params.number_of_experts,
                                        moe_params.intermediate_size};
    inType = outType = ov::element::f32;

    auto itr = configuration.find(ov::hint::inference_precision.name());
    if (itr != configuration.end() && itr->second == ov::element::bf16) {
        rel_threshold = 0.1f;
        abs_threshold = 0.1f;
        inType = outType = ov::element::bf16;
    }

    if (moe_type == MoEType::MoE2GeMM) {
        ASSERT_TRUE(activation_type == MoEActivationType::SWISH) << "MoE2GeMM only supports SWISH activation";
        function = initMoE2GeMMSubgraph(shape_params, ov::element::f32, ov::element::f32);
    } else if (moe_type == MoEType::MoE3GeMM) {
        function = initMoE3GeMMSubgraph(shape_params,
                                        ov::element::f32,
                                        ov::element::f32,
                                        false,
                                        std::nullopt,
                                        std::nullopt,
                                        std::nullopt,
                                        std::nullopt,
                                        std::nullopt,
                                        std::nullopt,
                                        MoERoutingType::SOFTMAX,
                                        activation_type);
    } else {
        OPENVINO_THROW("Unsupported MoEType");
    }
}

void MoESubgraphTest::check_results() {
    const auto& moe_type = std::get<1>(GetParam());
    const auto& gather_mm_nodes = get_gather_mm_nodes(compiledModel.get_runtime_model(), moe_type);
    const size_t expected_gather_mm_count = get_expected_gather_mm_count(moe_type);
    EXPECT_EQ(gather_mm_nodes.size(), expected_gather_mm_count);
}

std::string MoECompressedWeightsSubgraphTest::getTestCaseName(const testing::TestParamInfo<MoeCompressedWeightsTestParams>& obj) {
    const auto& [moe_params,
                    moe_type,
                    activation_type,
                    weights_precision,
                    decompression_precision,
                    scale_precision,
                    decompression_multiply_type,
                    decompression_subtract_type,
                    reshape_on_decompression,
                    decompression_group_size,
                    additional_config,
                    use_matmul_decompression_impl] = obj.param;
    std::ostringstream result;
    result << MoESubgraphTest::generateBaseMoeTestName(moe_params, moe_type, additional_config) << "_";
    result << "act=" << activation_type << "_";
    result << "_WP=" << weights_precision << "_";
    result << "DP=" << decompression_precision << "_";
    result << "SP=" << scale_precision << "_";
    result << "DM=" << decompression_multiply_type << "_";
    result << "DS=" << decompression_subtract_type << "_";
    result << "RD=" << reshape_on_decompression << "_";
    result << "GS=" << decompression_group_size << "_";
    result << "use_matmul_decompression_impl=" << use_matmul_decompression_impl << "_";
    return result.str();
}

void MoECompressedWeightsSubgraphTest::SetUp() {
    targetDevice = ov::test::utils::DEVICE_CPU;

    rel_threshold = 1e-3f;
    abs_threshold = 1e-3f;

    const auto& [moe_params,
                    moe_type,
                    activation_type,
                    weights_precision,
                    decompression_precision,
                    scale_precision,
                    decompression_multiply_type,
                    decompression_subtract_type,
                    reshape_on_decompression,
                    decompression_group_size,
                    additional_config,
                    use_matmul_decompression_impl] = GetParam();

    configuration.insert(additional_config.begin(), additional_config.end());
    init_input_shapes({moe_params.data_shape});
    const MoePatternParams shape_params{moe_params.data_shape.first,
                                        moe_params.topk,
                                        moe_params.number_of_experts,
                                        moe_params.intermediate_size};
    inType = outType = ov::element::f32;

    auto itr = configuration.find(ov::hint::inference_precision.name());
    if (itr != configuration.end() && itr->second == ov::element::bf16) {
        rel_threshold = 0.1f;
        abs_threshold = 0.1f;
        inType = outType = ov::element::bf16;
    }
        if(ov::with_cpu_arm_dotprod() || ov::with_cpu_arm_i8mm()){
        rel_threshold = 0.05f;
        abs_threshold = 0.05f;
        }

    if (moe_type == MoEType::MoE2GeMM) {
        ASSERT_TRUE(activation_type == MoEActivationType::SWISH) << "MoE2GeMM only supports SWISH activation";
        function = initMoE2GeMMSubgraph(shape_params,
                                        ov::element::f32,
                                        weights_precision,
                                        true,
                                        decompression_precision,
                                        scale_precision,
                                        decompression_multiply_type,
                                        decompression_subtract_type,
                                        reshape_on_decompression,
                                        decompression_group_size);
    } else if (moe_type == MoEType::MoE3GeMM) {
        function = initMoE3GeMMSubgraph(shape_params,
                                        ov::element::f32,
                                        weights_precision,
                                        true,
                                        decompression_precision,
                                        scale_precision,
                                        decompression_multiply_type,
                                        decompression_subtract_type,
                                        reshape_on_decompression,
                                        decompression_group_size,
                                        MoERoutingType::SOFTMAX,
                                        activation_type);
    } else {
        OPENVINO_THROW("Unsupported MoEType");
    }
}

void MoECompressedWeightsSubgraphTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    inputs.clear();
    const auto& itTargetShape = targetInputStaticShapes.front();
    const auto& params = function->get_parameters();
    ASSERT_EQ(params.size(), 1);
    auto param = params.front();
    auto type = param->get_element_type();

    auto input_tensor =
        ov::test::utils::create_and_fill_tensor(type,
                                                itTargetShape,
                                                ov::test::utils::InputGenerateData(0.125, 2, 8, 1234));

    inputs.insert({param, input_tensor});
}

void MoECompressedWeightsSubgraphTest::check_results() {
    const auto& test_param = GetParam();
    const auto& moe_type = std::get<1>(GetParam());
    const auto& gather_mm_nodes = MoESubgraphTest::get_gather_mm_nodes(compiledModel.get_runtime_model(), moe_type);
    const size_t expected_gather_mm_count = MoESubgraphTest::get_expected_gather_mm_count(moe_type);
    EXPECT_EQ(gather_mm_nodes.size(), expected_gather_mm_count);

    const ov::element::Type compressed_weights_precision = std::get<3>(test_param);
    const bool use_matmul_decompression_impl = std::get<11>(test_param);
    for (const auto& gather_mm_node : gather_mm_nodes) {
        const auto& expected_weights_precision = use_matmul_decompression_impl
                                                        ? compressed_weights_precision
                                                        : gather_mm_node->get_input_element_type(0);
        EXPECT_EQ(gather_mm_node->get_input_element_type(1), expected_weights_precision);
    }
}

TEST_P(MoESubgraphTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    #if defined(OPENVINO_ARCH_ARM)
        GTEST_SKIP();
    #endif
    run();
    check_results();
}

TEST_P(MoECompressedWeightsSubgraphTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    #if defined(OPENVINO_ARCH_ARM)
        GTEST_SKIP();
    #elif defined(OPENVINO_ARCH_ARM64)
        if (!ov::with_cpu_arm_dotprod() && !ov::with_cpu_arm_i8mm()){
            GTEST_SKIP();
        }
    #endif
    run();
    check_results();
}
}  // namespace ov::test
