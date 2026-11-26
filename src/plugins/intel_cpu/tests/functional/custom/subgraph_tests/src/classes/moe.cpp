// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "moe.hpp"

#include <cstring>
#include <memory>
#include <sstream>
#include <vector>

#include "common_test_utils/node_builders/moe_builders.hpp"
#include "common_test_utils/subgraph_builders/weights_decompression_builders.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "shared_test_classes/subgraph/weights_decompression_params.hpp"
#include "u3_symmetric_zero_point.hpp"
#include "utils/cpu_test_utils.hpp"
#include "utils/fusing_test_utils.hpp"

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
    // Quantizing the activations adds a small noise on top of u3's 8-level weights,
    // so such dynamic quantization path gets a slightly looser threshold.
    const bool is_bf16 = itr != configuration.end() && itr->second == ov::element::bf16;
    const auto dq_itr = configuration.find(ov::hint::dynamic_quantization_group_size.name());
    const uint64_t dq_group_size = dq_itr != configuration.end() ? dq_itr->second.as<uint64_t>() : 32;
    const bool dyn_quant =
        dq_group_size != 0 && dq_group_size % 32 == 0 &&
        (decompression_group_size == -1 || static_cast<uint64_t>(decompression_group_size) % dq_group_size == 0);
    if (weights_precision == ov::element::u3 && !is_bf16 && dyn_quant &&
        (ov::with_cpu_x86_avx2_vnni() || ov::with_cpu_x86_avx512_core_vnni())) {
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

namespace {
// One input in [0.125, 2.125) with a resolution of 1/8, the same values for every run.
void generate_small_range_inputs(std::map<std::shared_ptr<ov::Node>, ov::Tensor>& inputs,
                                 const std::shared_ptr<ov::Model>& function,
                                 const std::vector<ov::Shape>& targetInputStaticShapes) {
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
}  // namespace

namespace {
// One input whose element (row r, column k) is (0.125 + ((29 * k + 7 * r) % 17) / 8) * (0.25 + ((5 * r) % 13) / 8):
// positive like the small-range inputs, but every row has its own scale (0.25 to 1.75) and column pattern. Rows 64
// apart differ in scale by at least 1.55 times and rows 128 apart by at least 1.27 times.
void generate_row_pattern_inputs(std::map<std::shared_ptr<ov::Node>, ov::Tensor>& inputs,
                                 const std::shared_ptr<ov::Model>& function,
                                 const std::vector<ov::Shape>& targetInputStaticShapes) {
    inputs.clear();
    const auto& shape = targetInputStaticShapes.front();
    const auto& params = function->get_parameters();
    ASSERT_EQ(params.size(), 1);
    const auto& param = params.front();
    const auto type = param->get_element_type();
    ASSERT_TRUE(type == ov::element::f32 || type == ov::element::bf16) << "unexpected input type " << type;

    ov::Tensor input_tensor(type, shape);
    const size_t K = shape.back();
    const size_t rows = ov::shape_size(shape) / K;
    for (size_t r = 0; r < rows; r++) {
        for (size_t k = 0; k < K; k++) {
            const auto value = (0.125F + static_cast<float>((29 * k + 7 * r) % 17) / 8.0F) *
                               (0.25F + static_cast<float>((5 * r) % 13) / 8.0F);
            if (type == ov::element::f32) {
                input_tensor.data<float>()[r * K + k] = value;
            } else {
                input_tensor.data<ov::bfloat16>()[r * K + k] = ov::bfloat16(value);
            }
        }
    }
    inputs.insert({param, input_tensor});
}
}  // namespace

void MoESubgraphRowPatternTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    generate_row_pattern_inputs(inputs, function, targetInputStaticShapes);
}

void MoECompressedWeightsRowPatternTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    generate_row_pattern_inputs(inputs, function, targetInputStaticShapes);
}

void MoESubgraphSmallInputsTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    generate_small_range_inputs(inputs, function, targetInputStaticShapes);
}

void MoECompressedWeightsSubgraphTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    generate_small_range_inputs(inputs, function, targetInputStaticShapes);
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

TEST_P(MoESubgraphSmallInputsTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
#if defined(OPENVINO_ARCH_ARM)
    GTEST_SKIP();
#elif defined(OPENVINO_ARCH_ARM64)
    if (!ov::with_cpu_arm_dotprod() && !ov::with_cpu_arm_i8mm()) {
        GTEST_SKIP();
    }
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

void MoECompressedWeightsU3SymmetricZpSubgraphTest::SetUp() {
    MoECompressedWeightsSubgraphTest::SetUp();
    make_u3_symmetric_zero_points(function);
}

TEST_P(MoECompressedWeightsU3SymmetricZpSubgraphTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
    check_results();
}

TEST_P(MoESubgraphRowPatternTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
#if defined(OPENVINO_ARCH_ARM)
    GTEST_SKIP();
#endif
    run();
    check_results();
}

TEST_P(MoECompressedWeightsRowPatternTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
#if defined(OPENVINO_ARCH_ARM)
    GTEST_SKIP();
#elif defined(OPENVINO_ARCH_ARM64)
    if (!ov::with_cpu_arm_dotprod() && !ov::with_cpu_arm_i8mm()) {
        GTEST_SKIP();
    }
#endif
    run();
    check_results();
}

namespace {
// One infer request per target shape. Each request first runs alone, then all of them start at the same time
// for several rounds. A request must return, bit for bit, what it returned when it ran alone. The requests
// share the weights but nothing that a GatherMatmul executor writes while it executes, so what one request
// does (another input length, other GEMM heights, another task layout) must not show in another one's output.
void run_requests_in_parallel(ov::CompiledModel& compiled_model,
                              const std::vector<std::vector<ov::Shape>>& target_static_shapes) {
    const int32_t streams = compiled_model.get_property(ov::num_streams);
    ASSERT_GT(streams, 1) << "the model must be compiled with more than one stream for requests to overlap";

    struct Job {
        ov::Shape shape;
        ov::InferRequest request;
        ov::Tensor input;
        ov::Tensor alone;
    };
    std::vector<Job> jobs;
    const auto input_type = compiled_model.input().get_element_type();
    for (size_t i = 0; i < target_static_shapes.size(); ++i) {
        Job job;
        job.shape = target_static_shapes[i].front();
        job.request = compiled_model.create_infer_request();
        job.input = ov::test::utils::create_and_fill_tensor(
            input_type,
            job.shape,
            ov::test::utils::InputGenerateData(0.125, 2, 8, static_cast<int>(1234 + i)));
        job.request.set_input_tensor(job.input);
        job.request.infer();
        const auto output = job.request.get_output_tensor();
        job.alone = ov::Tensor(output.get_element_type(), output.get_shape());
        output.copy_to(job.alone);
        jobs.push_back(std::move(job));
    }
    ASSERT_GT(jobs.size(), static_cast<size_t>(streams)) << "more requests than streams, so that some also queue";

    constexpr size_t rounds = 8;
    for (size_t round = 0; round < rounds; ++round) {
        for (auto& job : jobs) {
            job.request.start_async();
        }
        for (auto& job : jobs) {
            job.request.wait();
        }
        for (auto& job : jobs) {
            const auto output = job.request.get_output_tensor();
            ASSERT_EQ(output.get_shape(), job.alone.get_shape());
            EXPECT_EQ(std::memcmp(output.data(), job.alone.data(), job.alone.get_byte_size()), 0)
                << "the request with input " << job.shape << " returned a different output in round " << round << " of "
                << jobs.size() << " requests on " << streams << " streams than when it ran alone";
        }
    }
}
}  // namespace

TEST_P(MoEParallelRequestsTest, RequestsInParallelReturnWhatTheyReturnAlone) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
#if defined(OPENVINO_ARCH_ARM)
    GTEST_SKIP();
#endif
    run();
    run_requests_in_parallel(compiledModel, targetStaticShapes);
}

TEST_P(MoECompressedWeightsParallelRequestsTest, RequestsInParallelReturnWhatTheyReturnAlone) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
#if defined(OPENVINO_ARCH_ARM)
    GTEST_SKIP();
#elif defined(OPENVINO_ARCH_ARM64)
    if (!ov::with_cpu_arm_dotprod() && !ov::with_cpu_arm_i8mm()) {
        GTEST_SKIP();
    }
#endif
    run();
    run_requests_in_parallel(compiledModel, targetStaticShapes);
}
}  // namespace ov::test
