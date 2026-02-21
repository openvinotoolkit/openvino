#include <gtest/gtest.h>

#include <memory>

#include "common_test_utils/test_common.hpp"
#include "cpu_streams_calculation.hpp"
#include "cpu_streams_calculation_prefer_threads.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/runtime/performance_heuristics.hpp"
#include "openvino/runtime/threading/cpu_streams_info.hpp"

using namespace testing;
using namespace ov;
using namespace ov::intel_cpu;
using namespace ov::intel_cpu::ThreadPreferenceConstants;

#if defined(OPENVINO_ARCH_ARM) && defined(__linux__)
using ov::intel_cpu::configure_arm_linux_threads;
#endif
#if (defined(OPENVINO_ARCH_ARM) || defined(OPENVINO_ARCH_ARM64)) && defined(__APPLE__)
using ov::intel_cpu::configure_apple_threads;
#endif
using ov::intel_cpu::configure_x86_hybrid_threads;
using ov::intel_cpu::configure_x86_non_hybrid_threads;
using ov::intel_cpu::configure_x86_throughput_threads;

namespace {

static std::shared_ptr<ov::Model> make_dummy_model(bool with_fakequantize = false) {
    using namespace ov::opset8;
    auto input = std::make_shared<Parameter>(ov::element::f32, ov::Shape{1, 3, 16, 16});
    std::shared_ptr<ov::Node> data = input;
    if (with_fakequantize) {
        auto fq = std::make_shared<FakeQuantize>(
            data,
            std::make_shared<Constant>(ov::element::f32, ov::Shape{1}, std::vector<float>{0}),
            std::make_shared<Constant>(ov::element::f32, ov::Shape{1}, std::vector<float>{1}),
            std::make_shared<Constant>(ov::element::f32, ov::Shape{1}, std::vector<float>{0}),
            std::make_shared<Constant>(ov::element::f32, ov::Shape{1}, std::vector<float>{1}),
            256);
        data = fq;
    }
    auto weights =
        std::make_shared<Constant>(ov::element::f32, ov::Shape{8, 3, 3, 3}, std::vector<float>(8 * 3 * 3 * 3, 1.0f));
    auto conv = std::make_shared<Convolution>(data,
                                              weights,
                                              ov::Strides{1, 1},
                                              ov::CoordinateDiff{0, 0},
                                              ov::CoordinateDiff{0, 0},
                                              ov::Strides{1, 1});
    auto result = std::make_shared<Result>(conv);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input});
}

class ModelPreferThreadsIntegrationTest : public ov::test::TestsCommon {};

TEST_F(ModelPreferThreadsIntegrationTest, INT8_Model_UseAllCores) {
    std::vector<std::vector<int>> proc_type_table = {{10, 2, 8, 0, 0, 0, 0}};
    auto model = make_dummy_model(true);
    Config config;
    config.modelType = Config::ModelType::CNN;
    config.modelPreferThreads = -1;
    int num_streams = 1;
    int result = get_model_prefer_threads(num_streams, proc_type_table, model, config);
    EXPECT_GE(config.modelPreferThreadsLatency, 10);
    EXPECT_EQ(result, config.modelPreferThreadsLatency);
}

TEST_F(ModelPreferThreadsIntegrationTest, LLM_Model_ECoresRatio) {
    std::vector<std::vector<int>> proc_type_table = {{14, 4, 10, 0, 0, 0, 0, 0}};
    auto model = make_dummy_model();
    Config config;
    config.modelType = Config::ModelType::LLM;
    config.modelPreferThreads = -1;
    int num_streams = 1;
    int result = get_model_prefer_threads(num_streams, proc_type_table, model, config);
    EXPECT_GE(config.modelPreferThreadsLatency, 14);
    EXPECT_EQ(result, config.modelPreferThreadsLatency);
}

TEST_F(ModelPreferThreadsIntegrationTest, ZeroCoresEdgeCase) {
    std::vector<std::vector<int>> proc_type_table = {{0, 0, 0, 0, 0, 0, 0}};
    auto model = make_dummy_model();
    Config config;
    config.modelType = Config::ModelType::CNN;
    config.modelPreferThreads = -1;
    int num_streams = 1;
    int result = get_model_prefer_threads(num_streams, proc_type_table, model, config);
    EXPECT_EQ(config.modelPreferThreadsLatency, 0);
    EXPECT_EQ(result, 0);
}

TEST_F(ModelPreferThreadsIntegrationTest, UnknownMemToleranceEdgeCase) {
    std::vector<std::vector<int>> proc_type_table = {{8, 8, 0, 0, 0, 0, 0, 0}};
    auto model = make_dummy_model();
    Config config;
    config.modelType = Config::ModelType::CNN;
    config.modelPreferThreads = -1;
    int num_streams = 0;
    int result = get_model_prefer_threads(num_streams, proc_type_table, model, config);
    EXPECT_GE(config.modelPreferThreadsThroughput, 1);
    EXPECT_EQ(result, config.modelPreferThreadsThroughput);
}

#if defined(OPENVINO_ARCH_ARM64) && defined(__linux__)
TEST_F(ModelPreferThreadsIntegrationTest, ARM64_Linux_DefaultThreads) {
    std::vector<std::vector<int>> proc_type_table = {{8, 8, 0, 0, 0, 0, 0}};
    auto model = make_dummy_model();
    Config config;
    config.modelType = Config::ModelType::CNN;
    config.modelPreferThreads = -1;
    int num_streams = 1;
    int result = get_model_prefer_threads(num_streams, proc_type_table, model, config);
    EXPECT_EQ(config.modelPreferThreadsLatency, 8);
    EXPECT_EQ(result, config.modelPreferThreadsLatency);
}
#endif

TEST_F(ModelPreferThreadsIntegrationTest, NumStreamsVsSocketsBoundary) {
    std::vector<std::vector<int>> proc_type_table = {{8, 8, 0, 0, 0, 0, 0}};
    auto model = make_dummy_model();
    Config config;
    config.modelType = Config::ModelType::CNN;
    config.modelPreferThreads = -1;
    int num_streams = 0;  // throughput path
    int result = get_model_prefer_threads(num_streams, proc_type_table, model, config);
    EXPECT_EQ(result, config.modelPreferThreadsThroughput);
    // Ensure the second case actually exercises the "num_streams > sockets" branch
    int sockets = get_num_sockets();
    num_streams = sockets + 1;  // guarantee num_streams > sockets
    result = get_model_prefer_threads(num_streams, proc_type_table, model, config);
    EXPECT_EQ(result, config.modelPreferThreadsThroughput);
}

class ModelPreferThreadsHelperTest : public ov::test::TestsCommon {};
class MainCoreCaseTest : public ov::test::TestsCommon {};
class StaticPartitionerCaseTest : public ov::test::TestsCommon {};
class TbbPartitionerDecisionTest : public ov::test::TestsCommon {};

TEST_F(ModelPreferThreadsHelperTest, ShouldUseAllCores_INT8_FewMainCores) {
    EXPECT_TRUE(should_use_all_cores_for_latency(2, 8, true));
    EXPECT_TRUE(should_use_all_cores_for_latency(1, 8, true));
    EXPECT_FALSE(should_use_all_cores_for_latency(3, 8, true));
}

TEST_F(ModelPreferThreadsHelperTest, ShouldUseAllCores_FP32_FewMainCores) {
    EXPECT_TRUE(should_use_all_cores_for_latency(4, 8, false));
    EXPECT_TRUE(should_use_all_cores_for_latency(3, 8, false));
    EXPECT_FALSE(should_use_all_cores_for_latency(5, 8, false));
}

TEST_F(ModelPreferThreadsHelperTest, ShouldUseAllCores_ManyMainCores) {
    EXPECT_FALSE(should_use_all_cores_for_latency(8, 4, true));
    EXPECT_FALSE(should_use_all_cores_for_latency(8, 4, false));
}

TEST_F(ModelPreferThreadsHelperTest, ShouldUseAllCores_NoEcores) {
    EXPECT_FALSE(should_use_all_cores_for_latency(4, 0, true));
    EXPECT_FALSE(should_use_all_cores_for_latency(4, 0, false));
}

TEST_F(ModelPreferThreadsHelperTest, ShouldUseEcoresForLLM_ManyEcores) {
    EXPECT_TRUE(should_use_ecores_for_llm(10, 4));
    EXPECT_TRUE(should_use_ecores_for_llm(9, 4));
    EXPECT_FALSE(should_use_ecores_for_llm(8, 4));
    EXPECT_FALSE(should_use_ecores_for_llm(6, 4));
}

TEST_F(ModelPreferThreadsHelperTest, ShouldUseEcoresForLLM_FewEcores) {
    EXPECT_FALSE(should_use_ecores_for_llm(4, 8));
    EXPECT_FALSE(should_use_ecores_for_llm(0, 8));
}

TEST_F(ModelPreferThreadsHelperTest, IsNetworkComputeLimited_AllComputeConvs) {
    ov::MemBandwidthPressure tolerance;
    tolerance.ratio_compute_convs = ov::MemBandwidthPressure::ALL;
    tolerance.ratio_compute_deconvs = 0.0f;
    EXPECT_TRUE(is_network_compute_limited(tolerance));
}

TEST_F(ModelPreferThreadsHelperTest, IsNetworkComputeLimited_AllComputeDeconvs) {
    ov::MemBandwidthPressure tolerance;
    tolerance.ratio_compute_convs = 0.0f;
    tolerance.ratio_compute_deconvs = ov::MemBandwidthPressure::ALL;
    EXPECT_TRUE(is_network_compute_limited(tolerance));
}

TEST_F(ModelPreferThreadsHelperTest, IsNetworkComputeLimited_PartialCompute) {
    ov::MemBandwidthPressure tolerance;
    tolerance.ratio_compute_convs = 0.5f;
    tolerance.ratio_compute_deconvs = 0.3f;
    EXPECT_FALSE(is_network_compute_limited(tolerance));
}

TEST_F(ModelPreferThreadsHelperTest, IsBelowIsaThreshold) {
    EXPECT_FALSE(is_below_isa_threshold(0.6f, 1.0f));
    EXPECT_FALSE(is_below_isa_threshold(0.4f, 1.0f));
    EXPECT_FALSE(is_below_isa_threshold(0.3f, 2.0f));
    EXPECT_FALSE(is_below_isa_threshold(0.2f, 2.0f));
    EXPECT_TRUE(is_below_isa_threshold(1.1f, 1.0f));
    EXPECT_TRUE(is_below_isa_threshold(2.1f, 2.0f));
}

TEST_F(ModelPreferThreadsHelperTest, IsBelowGeneralThreshold) {
    EXPECT_TRUE(is_below_general_threshold(0.6f));
    EXPECT_FALSE(is_below_general_threshold(0.4f));
    EXPECT_FALSE(is_below_general_threshold(0.5f));
}

TEST_F(MainCoreCaseTest, MainCoreCase1_HighMemLimitedConvs) {
    ov::MemBandwidthPressure tolerance;
    tolerance.ratio_mem_limited_convs = 0.85f;
    EXPECT_TRUE(is_main_core_case_1(tolerance));
    tolerance.ratio_mem_limited_convs = 0.8f;
    EXPECT_FALSE(is_main_core_case_1(tolerance));
    tolerance.ratio_mem_limited_convs = 0.5f;
    EXPECT_FALSE(is_main_core_case_1(tolerance));
}

TEST_F(MainCoreCaseTest, MainCoreCase2_NoConvsHighTolerance) {
    ov::MemBandwidthPressure tolerance;
    tolerance.ratio_mem_limited_convs = 0.0f;
    tolerance.ratio_compute_convs = 0.0f;
    tolerance.max_mem_tolerance = 5.0f;
    EXPECT_TRUE(is_main_core_case_2(tolerance));
    tolerance.max_mem_tolerance = 4.5f;
    EXPECT_TRUE(is_main_core_case_2(tolerance));
    tolerance.max_mem_tolerance = 4.0f;
    EXPECT_FALSE(is_main_core_case_2(tolerance));
    tolerance.max_mem_tolerance = 5.0f;
    tolerance.ratio_compute_convs = 0.1f;
    EXPECT_FALSE(is_main_core_case_2(tolerance));
    tolerance.ratio_compute_convs = 0.0f;
    tolerance.ratio_mem_limited_convs = 0.1f;
    EXPECT_FALSE(is_main_core_case_2(tolerance));
}

TEST_F(MainCoreCaseTest, MainCoreCase3_MostlyLightConvs) {
    ov::MemBandwidthPressure tolerance;
    tolerance.ratio_mem_limited_convs = 0.0f;
    tolerance.ratio_compute_convs = 0.5f;
    tolerance.total_convs = 100;
    tolerance.total_light_convs = 95;
    EXPECT_TRUE(is_main_core_case_3(tolerance));
    tolerance.total_light_convs = 90;
    EXPECT_FALSE(is_main_core_case_3(tolerance));
    tolerance.total_light_convs = 95;
    tolerance.ratio_compute_convs = 1.0f;
    EXPECT_FALSE(is_main_core_case_3(tolerance));
    tolerance.ratio_compute_convs = 0.0f;
    EXPECT_FALSE(is_main_core_case_3(tolerance));
    tolerance.ratio_compute_convs = 0.5f;
    tolerance.ratio_mem_limited_convs = 0.1f;
    EXPECT_FALSE(is_main_core_case_3(tolerance));
}

TEST_F(MainCoreCaseTest, MainCoreCase4_MixedWorkload) {
    ov::MemBandwidthPressure tolerance;
    tolerance.ratio_mem_limited_convs = 0.3f;
    tolerance.ratio_compute_convs = 0.4f;
    tolerance.total_convs = 100;
    tolerance.total_light_convs = 50;
    EXPECT_TRUE(is_main_core_case_4(tolerance));
    tolerance.total_light_convs = 46;
    EXPECT_FALSE(is_main_core_case_4(tolerance));
    tolerance.total_light_convs = 50;
    tolerance.ratio_mem_limited_convs = 0.0f;
    EXPECT_FALSE(is_main_core_case_4(tolerance));
    tolerance.ratio_mem_limited_convs = 0.3f;
    tolerance.ratio_compute_convs = 0.0f;
    EXPECT_FALSE(is_main_core_case_4(tolerance));
}

TEST_F(StaticPartitionerCaseTest, StaticCase1_NoNodes) {
    ov::MemBandwidthPressure tolerance;
    tolerance.total_nodes = 0;
    EXPECT_TRUE(is_static_partitioner_case_1(tolerance));
    tolerance.total_nodes = 1;
    EXPECT_FALSE(is_static_partitioner_case_1(tolerance));
}

TEST_F(StaticPartitionerCaseTest, StaticCase2_MajorityLightConvs) {
    ov::MemBandwidthPressure tolerance;
    tolerance.total_convs = 100;
    tolerance.total_light_convs = 65;
    EXPECT_TRUE(is_static_partitioner_case_2(tolerance));
    tolerance.total_light_convs = 60;
    EXPECT_FALSE(is_static_partitioner_case_2(tolerance));
    tolerance.total_convs = 0;
    tolerance.total_light_convs = 0;
    EXPECT_FALSE(is_static_partitioner_case_2(tolerance));
}

TEST_F(StaticPartitionerCaseTest, StaticCase3_WithLpEcores) {
    ov::MemBandwidthPressure tolerance;
    tolerance.total_convs = 100;
    tolerance.total_light_convs = 50;
    tolerance.ratio_compute_convs = 0.3f;
    tolerance.ratio_mem_limited_convs = 0.1f;
    tolerance.ratio_mem_limited_gemms = 0.0f;
    tolerance.ratio_mem_limited_adds = 0.2f;
    tolerance.max_mem_tolerance = 0.1f;
    EXPECT_TRUE(is_static_partitioner_case_3_with_lp_ecores(tolerance));
    tolerance.ratio_mem_limited_convs = 0.25f;
    EXPECT_FALSE(is_static_partitioner_case_3_with_lp_ecores(tolerance));
}

TEST_F(StaticPartitionerCaseTest, StaticCase3_WithoutLpEcores) {
    ov::MemBandwidthPressure tolerance;
    tolerance.total_convs = 100;
    tolerance.total_light_convs = 50;
    tolerance.ratio_compute_convs = 0.3f;
    tolerance.ratio_mem_limited_convs = 0.1f;
    tolerance.ratio_mem_limited_gemms = 0.0f;
    tolerance.ratio_mem_limited_adds = 0.2f;
    tolerance.max_mem_tolerance = 0.1f;
    EXPECT_TRUE(is_static_partitioner_case_3_without_lp_ecores(tolerance));
    tolerance.max_mem_tolerance = 0.05f;
    EXPECT_FALSE(is_static_partitioner_case_3_without_lp_ecores(tolerance));
}

TEST_F(StaticPartitionerCaseTest, StaticCase4_WithLpEcores) {
    ov::MemBandwidthPressure tolerance;
    tolerance.total_convs = 0;
    tolerance.max_mem_tolerance = 3.0f;
    tolerance.total_gemms = 0;
    tolerance.total_nodes = 100;
    EXPECT_TRUE(is_static_partitioner_case_4_with_lp_ecores(tolerance));
    tolerance.max_mem_tolerance = 1.0f;
    tolerance.total_gemms = 15;
    EXPECT_TRUE(is_static_partitioner_case_4_with_lp_ecores(tolerance));
    tolerance.total_convs = 10;
    EXPECT_FALSE(is_static_partitioner_case_4_with_lp_ecores(tolerance));
}

TEST_F(StaticPartitionerCaseTest, StaticCase4_WithoutLpEcores) {
    ov::MemBandwidthPressure tolerance;
    tolerance.total_convs = 0;
    tolerance.total_gemms = 4;
    tolerance.total_nodes = 100;
    EXPECT_TRUE(is_static_partitioner_case_4_without_lp_ecores(tolerance));
    tolerance.total_gemms = 6;
    EXPECT_FALSE(is_static_partitioner_case_4_without_lp_ecores(tolerance));
}

TEST_F(StaticPartitionerCaseTest, StaticCase5_OnlyWithLpEcores) {
    ov::MemBandwidthPressure tolerance;
    tolerance.total_convs = 100;
    tolerance.total_light_convs = 50;
    tolerance.ratio_compute_convs = 1.0f;
    tolerance.ratio_mem_limited_convs = 0.5f;
    tolerance.ratio_mem_limited_adds = 1.0f;
    tolerance.total_heavy_convs = 15;
    tolerance.total_nodes = 100;
    EXPECT_TRUE(is_static_partitioner_case_5(tolerance));
    tolerance.ratio_compute_convs = 0.9f;
    EXPECT_FALSE(is_static_partitioner_case_5(tolerance));
}

TEST_F(ModelPreferThreadsHelperTest, IsLpMainCoreCase1_TrueAndFalse) {
    ov::MemBandwidthPressure tolerance;
    tolerance.total_convs = 0;
    tolerance.max_mem_tolerance = MEM_TOLERANCE_VERY_HIGH + 1.0f;
    tolerance.total_nodes = 100;
    tolerance.total_gemms = 4;
    EXPECT_TRUE(is_lp_main_core_case_1(tolerance));
    tolerance.total_gemms = 6;
    EXPECT_FALSE(is_lp_main_core_case_1(tolerance));
    tolerance.total_gemms = 4;
    tolerance.max_mem_tolerance = MEM_TOLERANCE_VERY_HIGH;
    EXPECT_FALSE(is_lp_main_core_case_1(tolerance));
    tolerance.max_mem_tolerance = MEM_TOLERANCE_VERY_HIGH + 0.1f;
    tolerance.total_convs = 1;
    EXPECT_FALSE(is_lp_main_core_case_1(tolerance));
}

TEST_F(ModelPreferThreadsHelperTest, IsLpMainCoreCase2_TrueAndFalse) {
    ov::MemBandwidthPressure tolerance;

    tolerance.total_convs = 10;
    tolerance.total_gemms = 1;
    tolerance.max_mem_tolerance = MEM_TOLERANCE_MEDIUM_LOW - 0.1f;
    tolerance.total_light_convs = 9;
    EXPECT_TRUE(is_lp_main_core_case_2(tolerance));
    tolerance.total_light_convs = 7;
    EXPECT_FALSE(is_lp_main_core_case_2(tolerance));
    tolerance.total_light_convs = 9;
    tolerance.total_gemms = 2;
    EXPECT_FALSE(is_lp_main_core_case_2(tolerance));
    tolerance.total_gemms = 1;
    tolerance.max_mem_tolerance = MEM_TOLERANCE_MEDIUM_LOW + 0.1f;
    EXPECT_FALSE(is_lp_main_core_case_2(tolerance));
}

TEST_F(ModelPreferThreadsIntegrationTest, Configure_X86_Hybrid_LP_UsesMainOrMainPlusLp) {
    std::vector<std::vector<int>> proc_type_table = {{6, 2, 0, 4, 0, 0, 0}};
    Config cfg1;
    ov::MemBandwidthPressure tol1;
    tol1.total_convs = 0;
    tol1.max_mem_tolerance = MEM_TOLERANCE_VERY_HIGH + 2.0f;
    tol1.total_nodes = 100;
    tol1.total_gemms = 2;
    configure_x86_hybrid_lp_threads(cfg1, proc_type_table, tol1);
    EXPECT_EQ(cfg1.modelPreferThreadsLatency, 2);

    Config cfg2;
    ov::MemBandwidthPressure tol2;
    tol2.total_convs = 10;
    tol2.total_gemms = 1;
    tol2.max_mem_tolerance = MEM_TOLERANCE_MEDIUM_LOW + 0.2f;
    tol2.total_light_convs = 5;
    configure_x86_hybrid_lp_threads(cfg2, proc_type_table, tol2);
    EXPECT_EQ(cfg2.modelPreferThreadsLatency, 2 + 4);
}

TEST_F(TbbPartitionerDecisionTest, AlreadyConfigured) {
    Config config;
    config.tbbPartitioner = TbbPartitioner::STATIC;
    std::vector<std::vector<int>> proc_type_table = {{20, 6, 8, 4, 0, 0, 0}};
    ov::MemBandwidthPressure tolerance;
    determine_tbb_partitioner_and_threads(config, proc_type_table, tolerance, true);
    EXPECT_EQ(TbbPartitioner::STATIC, config.tbbPartitioner);
}

TEST_F(TbbPartitionerDecisionTest, MainCoreCase_WithLpEcores_INT8) {
    Config config;
    config.tbbPartitioner = TbbPartitioner::NONE;
    std::vector<std::vector<int>> proc_type_table = {{20, 6, 8, 4, 0, 0, 0}};
    ov::MemBandwidthPressure tolerance;
    tolerance.total_convs = 100;
    tolerance.ratio_mem_limited_convs = 0.85f;
    determine_tbb_partitioner_and_threads(config, proc_type_table, tolerance, true);
    EXPECT_EQ(TbbPartitioner::STATIC, config.tbbPartitioner);
    EXPECT_EQ(6, config.modelPreferThreadsLatency);
}

TEST_F(TbbPartitionerDecisionTest, StaticPartitioner_NoNodes) {
    Config config;
    config.tbbPartitioner = TbbPartitioner::NONE;
    std::vector<std::vector<int>> proc_type_table = {{20, 6, 8, 0, 0, 0, 0}};
    ov::MemBandwidthPressure tolerance;
    tolerance.total_nodes = 0;
    determine_tbb_partitioner_and_threads(config, proc_type_table, tolerance, false);
    EXPECT_EQ(TbbPartitioner::STATIC, config.tbbPartitioner);
}

TEST_F(TbbPartitionerDecisionTest, AutoPartitioner_Default) {
    Config config;
    config.tbbPartitioner = TbbPartitioner::NONE;
    std::vector<std::vector<int>> proc_type_table = {{20, 6, 8, 0, 0, 0, 0}};
    ov::MemBandwidthPressure tolerance;
    tolerance.total_nodes = 100;
    tolerance.total_convs = 100;
    tolerance.total_light_convs = 30;
    tolerance.ratio_compute_convs = 0.8f;
    tolerance.ratio_mem_limited_convs = 0.15f;
    tolerance.ratio_mem_limited_gemms = 0.1f;
    tolerance.total_gemms = 10;
    determine_tbb_partitioner_and_threads(config, proc_type_table, tolerance, false);
    EXPECT_EQ(TbbPartitioner::AUTO, config.tbbPartitioner);
}

TEST_F(ModelPreferThreadsHelperTest, GetIsaThresholdMultiplier) {
    EXPECT_FLOAT_EQ(ISA_THRESHOLD_SSE41, get_isa_threshold_multiplier(dnnl::cpu_isa::sse41));
    EXPECT_FLOAT_EQ(ISA_THRESHOLD_AVX2, get_isa_threshold_multiplier(dnnl::cpu_isa::avx2));
    EXPECT_FLOAT_EQ(ISA_THRESHOLD_AVX2, get_isa_threshold_multiplier(dnnl::cpu_isa::avx512_core));
    EXPECT_FLOAT_EQ(ISA_THRESHOLD_VNNI, get_isa_threshold_multiplier(dnnl::cpu_isa::avx512_core_vnni));
    EXPECT_FLOAT_EQ(ISA_THRESHOLD_VNNI, get_isa_threshold_multiplier(dnnl::cpu_isa::avx2_vnni));
    EXPECT_FLOAT_EQ(ISA_THRESHOLD_VNNI, get_isa_threshold_multiplier(dnnl::cpu_isa::avx2_vnni_2));
    EXPECT_FLOAT_EQ(ISA_THRESHOLD_AMX, get_isa_threshold_multiplier(dnnl::cpu_isa::avx512_core_amx));
}

TEST_F(ModelPreferThreadsHelperTest, EdgeCase_ZeroConvs) {
    ov::MemBandwidthPressure tolerance;
    tolerance.total_convs = 0;
    tolerance.total_light_convs = 0;
    EXPECT_FALSE(is_main_core_case_3(tolerance));
    EXPECT_FALSE(is_main_core_case_4(tolerance));
    EXPECT_FALSE(is_static_partitioner_case_2(tolerance));
}

TEST_F(ModelPreferThreadsHelperTest, EdgeCase_ZeroNodes) {
    ov::MemBandwidthPressure tolerance;
    tolerance.total_nodes = 0;
    tolerance.total_gemms = 0;
    EXPECT_TRUE(is_static_partitioner_case_1(tolerance));
}

TEST_F(ModelPreferThreadsHelperTest, EdgeCase_UnknownTolerance) {
    ov::MemBandwidthPressure tolerance;
    tolerance.max_mem_tolerance = ov::MemBandwidthPressure::UNKNOWN;
    EXPECT_TRUE(is_main_core_case_2(tolerance));
    EXPECT_TRUE(is_below_isa_threshold(tolerance.max_mem_tolerance, 1.0f));
    EXPECT_TRUE(is_below_general_threshold(tolerance.max_mem_tolerance));
}

TEST_F(ModelPreferThreadsHelperTest, BoundaryValues_ConvRatios) {
    ov::MemBandwidthPressure tolerance;
    tolerance.total_convs = 100;
    tolerance.ratio_mem_limited_convs = CONV_RATIO_HIGH;
    EXPECT_FALSE(is_main_core_case_1(tolerance));
    tolerance.ratio_mem_limited_convs = CONV_RATIO_HIGH + 0.001f;
    EXPECT_TRUE(is_main_core_case_1(tolerance));
    tolerance.total_light_convs = static_cast<int>(CONV_RATIO_MEDIUM * 100);
    EXPECT_FALSE(is_static_partitioner_case_2(tolerance));
    tolerance.total_light_convs = static_cast<int>(CONV_RATIO_MEDIUM * 100) + 1;
    EXPECT_TRUE(is_static_partitioner_case_2(tolerance));
}

TEST_F(ModelPreferThreadsHelperTest, BoundaryValues_MemTolerance) {
    ov::MemBandwidthPressure tolerance;
    tolerance.ratio_mem_limited_convs = 0.0f;
    tolerance.ratio_compute_convs = 0.0f;
    tolerance.max_mem_tolerance = MEM_TOLERANCE_HIGH;
    EXPECT_TRUE(is_main_core_case_2(tolerance));
    tolerance.max_mem_tolerance = MEM_TOLERANCE_HIGH - 0.001f;
    EXPECT_FALSE(is_main_core_case_2(tolerance));
}

TEST_F(ModelPreferThreadsHelperTest, ConstantsAreValid) {
    EXPECT_GT(INT8_EFFICIENCY_THRESHOLD, FP32_EFFICIENCY_THRESHOLD);
    EXPECT_GT(ISA_THRESHOLD_AMX, ISA_THRESHOLD_VNNI);
    EXPECT_GT(ISA_THRESHOLD_VNNI, ISA_THRESHOLD_AVX2);
    EXPECT_GT(ISA_THRESHOLD_AVX2, ISA_THRESHOLD_SSE41);
    EXPECT_GT(CONV_RATIO_VERY_HIGH, CONV_RATIO_HIGH);
    EXPECT_GT(CONV_RATIO_HIGH, CONV_RATIO_MEDIUM);
    EXPECT_GT(CONV_RATIO_MEDIUM, CONV_RATIO_LOW);
    EXPECT_GT(CONV_RATIO_LOW, CONV_RATIO_MINIMAL);
    EXPECT_GT(CONV_RATIO_MINIMAL, CONV_RATIO_VERY_LOW);
    EXPECT_GT(CONV_RATIO_VERY_LOW, CONV_RATIO_ULTRA_LOW);
    EXPECT_GT(MEM_TOLERANCE_HIGH, MEM_TOLERANCE_MEDIUM);
    EXPECT_GT(MEM_TOLERANCE_MEDIUM, MEM_TOLERANCE_MEDIUM_LOW);
    EXPECT_GT(MEM_TOLERANCE_MEDIUM_LOW, MEM_TOLERANCE_VERY_LOW);
}

TEST_F(ModelPreferThreadsHelperTest, IsLpAutoCase1_TrueAndFalse) {
    ov::MemBandwidthPressure tolerance;
    tolerance.total_convs = 60;
    tolerance.ratio_compute_convs = 0.1f;
    tolerance.ratio_mem_limited_convs = 0.01f;
    EXPECT_TRUE(is_lp_auto_case_1(tolerance));

    tolerance.ratio_mem_limited_convs = 0.3f;
    EXPECT_FALSE(is_lp_auto_case_1(tolerance));
}

TEST_F(ModelPreferThreadsHelperTest, IsLpAutoCase2_TrueAndFalse) {
    ov::MemBandwidthPressure tolerance;
    tolerance.max_mem_tolerance = MEM_TOLERANCE_SECONDARY_LOW - 0.01f;
    tolerance.ratio_compute_convs = 0.4f;
    tolerance.ratio_mem_limited_convs = 0.1f;
    EXPECT_TRUE(is_lp_auto_case_2(tolerance));

    tolerance.max_mem_tolerance = MEM_TOLERANCE_SECONDARY_LOW + 1.0f;
    EXPECT_FALSE(is_lp_auto_case_2(tolerance));
}

TEST_F(ModelPreferThreadsHelperTest, IsLpAutoCase3_TrueAndFalse) {
    ov::MemBandwidthPressure tolerance;
    tolerance.ratio_compute_convs = 0.05f;
    tolerance.ratio_mem_limited_convs = 0.2f;
    EXPECT_TRUE(is_lp_auto_case_3(tolerance));

    tolerance.ratio_compute_convs = 0.2f;
    EXPECT_FALSE(is_lp_auto_case_3(tolerance));
}

TEST_F(ModelPreferThreadsHelperTest, IsLpAutoCase4_TrueAndFalse) {
    ov::MemBandwidthPressure tolerance;
    tolerance.max_mem_tolerance = MEM_TOLERANCE_LOW + 1.0f;
    tolerance.ratio_compute_convs = CONV_RATIO_MEDIUM_LOW + 0.1f;
    tolerance.ratio_mem_limited_adds = 0.5f;
    tolerance.total_adds = 1;
    tolerance.total_nodes = 100;
    EXPECT_TRUE(is_lp_auto_case_4(tolerance));

    tolerance.ratio_compute_convs = 0.1f;
    EXPECT_FALSE(is_lp_auto_case_4(tolerance));
}

TEST_F(ModelPreferThreadsHelperTest, IsLpAutoCase5_TrueAndFalse) {
    ov::MemBandwidthPressure tolerance;
    tolerance.max_mem_tolerance = MEM_TOLERANCE_SECONDARY_LOW;
    tolerance.total_light_convs = 11;
    EXPECT_TRUE(is_lp_auto_case_5(tolerance));

    tolerance.total_light_convs = 5;
    EXPECT_FALSE(is_lp_auto_case_5(tolerance));
}

TEST_F(ModelPreferThreadsIntegrationTest, Direct_X86_NonHybrid_ZeroMain) {
    Config config;
    std::vector<std::vector<int>> proc_type_table = {{4, 0, 4, 0, 0, 0, 0}};
    configure_x86_non_hybrid_threads(config, proc_type_table);
    EXPECT_EQ(config.modelPreferThreadsLatency, 4);
}

TEST_F(ModelPreferThreadsIntegrationTest, Direct_X86_Hybrid_LLM_MainOnly) {
    Config config;
    std::vector<std::vector<int>> proc_type_table = {{13, 4, 8, 1, 0, 0, 0}};
    ov::MemBandwidthPressure tolerance;
    configure_x86_hybrid_threads(config, proc_type_table, tolerance, false, true);
    EXPECT_EQ(config.modelPreferThreadsLatency, 4);
}

TEST_F(ModelPreferThreadsIntegrationTest, Direct_X86_Hybrid_Fallback_UseAll) {
    Config config;
    config.threads = 0;  // allow hybrid-applicable fallback
    std::vector<std::vector<int>> proc_type_table = {{12, 4, 8, 0, 0, 0, 0}};
    ov::MemBandwidthPressure tolerance;
    configure_x86_hybrid_threads(config, proc_type_table, tolerance, false, false);
    EXPECT_EQ(config.modelPreferThreadsLatency, 12);
}

TEST_F(ModelPreferThreadsIntegrationTest, Direct_X86_Throughput_HT_Adjustment) {
    Config config;
    std::vector<std::vector<int>> proc_type_table = {{16, 8, 0, 0, 8, 0, 0}};
    ov::MemBandwidthPressure tolerance;
    tolerance.max_mem_tolerance = ov::MemBandwidthPressure::UNKNOWN;
    tolerance.ratio_compute_convs = ov::MemBandwidthPressure::ALL;
    EXPECT_TRUE(is_network_compute_limited(tolerance));
    configure_x86_throughput_threads(config, proc_type_table, tolerance, 1.0f);
    EXPECT_GE(config.modelPreferThreadsThroughput, 1);
}

#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)

TEST_F(ModelPreferThreadsIntegrationTest, X86_NonHybrid_ZeroMainUsesEfficient) {
    std::vector<std::vector<int>> proc_type_table = {{4, 0, 4, 0, 0, 0, 0}};
    auto model = make_dummy_model();
    Config config;
    config.modelType = Config::ModelType::CNN;
    config.modelPreferThreads = -1;
    int num_streams = 1;
    int result = get_model_prefer_threads(num_streams, proc_type_table, model, config);
    EXPECT_EQ(config.modelPreferThreadsLatency, 4);
    EXPECT_EQ(result, 4);
}

TEST_F(ModelPreferThreadsIntegrationTest, X86_Hybrid_HybridApplicable_IsLLM_MainOnly) {
    std::vector<std::vector<int>> proc_type_table = {{13, 4, 8, 4, 0, 0, 0}};
    auto model = make_dummy_model();
    Config config;
    config.modelType = Config::ModelType::LLM;
    config.modelPreferThreads = -1;
    int num_streams = 1;
    int result = get_model_prefer_threads(num_streams, proc_type_table, model, config);
    EXPECT_EQ(config.modelPreferThreadsLatency, 4);
    EXPECT_EQ(result, 4);
}

TEST_F(ModelPreferThreadsIntegrationTest, X86_Hybrid_Fallback_UseAllCores) {
    std::vector<std::vector<int>> proc_type_table = {{12, 4, 8, 0, 0, 0, 0}};
    auto model = make_dummy_model();
    Config config;
    config.modelType = Config::ModelType::CNN;
    config.threads = 1;
    config.modelPreferThreads = -1;
    int num_streams = 1;
    int result = get_model_prefer_threads(num_streams, proc_type_table, model, config);
    EXPECT_EQ(config.modelPreferThreadsLatency, 12);
    EXPECT_EQ(result, 12);
}

TEST_F(ModelPreferThreadsIntegrationTest, X86_Throughput_HyperThreadingAdjustment) {
    std::vector<std::vector<int>> proc_type_table = {{16, 8, 0, 0, 8, 0, 0}};
    auto model = make_dummy_model();
    Config config;
    config.modelType = Config::ModelType::CNN;
    config.modelPreferThreads = -1;
    int num_streams = 0;
    int result = get_model_prefer_threads(num_streams, proc_type_table, model, config);
    EXPECT_GE(config.modelPreferThreadsThroughput, 1);
    EXPECT_EQ(result, config.modelPreferThreadsThroughput);
}

TEST_F(ModelPreferThreadsIntegrationTest, X86_NonHybrid_MainCoresOnly) {
    std::vector<std::vector<int>> proc_type_table = {{8, 8, 0, 0, 0, 0, 0}};
    auto model = make_dummy_model();
    Config config;
    config.modelType = Config::ModelType::CNN;
    config.modelPreferThreads = -1;
    int num_streams = 1;
    int result = get_model_prefer_threads(num_streams, proc_type_table, model, config);
    EXPECT_EQ(config.modelPreferThreadsLatency, 8);
    EXPECT_EQ(result, config.modelPreferThreadsLatency);
}

TEST_F(ModelPreferThreadsIntegrationTest, X86_Hybrid_INT8_UseAllCores) {
    std::vector<std::vector<int>> proc_type_table = {{12, 4, 8, 0, 0, 0, 0}};
    auto model = make_dummy_model();
    Config config;
    config.modelType = Config::ModelType::CNN;
    config.modelPreferThreads = -1;
    int num_streams = 1;
    int result = get_model_prefer_threads(num_streams, proc_type_table, model, config);
    EXPECT_GE(config.modelPreferThreadsLatency, 12);
    EXPECT_EQ(result, config.modelPreferThreadsLatency);
}

TEST_F(ModelPreferThreadsIntegrationTest, X86_Hybrid_FP32_MainCoresOnly) {
    std::vector<std::vector<int>> proc_type_table = {{8, 4, 4, 0, 0, 0, 0}};
    auto model = make_dummy_model();
    Config config;
    config.modelType = Config::ModelType::CNN;
    config.modelPreferThreads = -1;
    int num_streams = 1;
    int result = get_model_prefer_threads(num_streams, proc_type_table, model, config);
    EXPECT_EQ(config.modelPreferThreadsLatency, 4);
    EXPECT_EQ(result, config.modelPreferThreadsLatency);
}

TEST_F(ModelPreferThreadsIntegrationTest, X86_HT_AdjustThroughput) {
    std::vector<std::vector<int>> proc_type_table = {{16, 8, 0, 0, 8, 0, 0}};
    auto model = make_dummy_model();
    Config config;
    config.modelType = Config::ModelType::CNN;
    config.modelPreferThreads = -1;
    int num_streams = 0;  // throughput path
    int result = get_model_prefer_threads(num_streams, proc_type_table, model, config);
    EXPECT_GE(config.modelPreferThreadsThroughput, 1);
    EXPECT_EQ(result, config.modelPreferThreadsThroughput);
}

#endif

#if defined(OPENVINO_ARCH_ARM) && defined(__linux__)
TEST_F(ModelPreferThreadsIntegrationTest, ARM_Linux_Throughput_UnknownAndMemLimited) {
    std::vector<std::vector<int>> proc_type_table = {{8, 4, 4, 0, 0, 0, 0}};
    auto model = make_dummy_model();
    Config config;
    config.modelType = Config::ModelType::CNN;
    config.modelPreferThreads = -1;

    int result = get_model_prefer_threads(0, proc_type_table, model, config);
    EXPECT_GE(config.modelPreferThreadsThroughput, 1);
    EXPECT_EQ(result, config.modelPreferThreadsThroughput);
}

TEST_F(ModelPreferThreadsIntegrationTest, Direct_ARM_Linux_ThroughputBranches) {
    Config config;
    std::vector<std::vector<int>> proc_type_table = {{8, 4, 4, 0, 0, 0, 0}};
    ov::MemBandwidthPressure tolerance;
    // UNKNOWN should allow high throughput when compute-limited
    tolerance.max_mem_tolerance = ov::MemBandwidthPressure::UNKNOWN;
    configure_arm_linux_threads(config, proc_type_table, tolerance, false, false);
    EXPECT_GE(config.modelPreferThreadsThroughput, 1);
}
#endif

#if (defined(OPENVINO_ARCH_ARM) || defined(OPENVINO_ARCH_ARM64)) && defined(__APPLE__)
TEST_F(ModelPreferThreadsIntegrationTest, AppleSilicon_SizeOne_SpecialLatencyChoice) {
    std::vector<std::vector<int>> proc_type_table = {{10, 6, 4, 0, 0, 0, 0}};
    auto model = make_dummy_model();
    Config config;
    config.modelType = Config::ModelType::CNN;
    config.modelPreferThreads = -1;
    int result = get_model_prefer_threads(1, proc_type_table, model, config);
    EXPECT_EQ(config.modelPreferThreadsLatency, 6);
    EXPECT_EQ(result, 6);
}

TEST_F(ModelPreferThreadsIntegrationTest, AppleSilicon_LatencyAndThroughput) {
    std::vector<std::vector<int>> proc_type_table = {{8, 4, 4, 0, 0, 0, 0}};
    auto model = make_dummy_model();
    Config config;
    config.modelType = Config::ModelType::CNN;
    config.modelPreferThreads = -1;
    int num_streams = 1;
    int result = get_model_prefer_threads(num_streams, proc_type_table, model, config);
    EXPECT_GE(config.modelPreferThreadsLatency, 4);
    EXPECT_EQ(result, config.modelPreferThreadsLatency);
}

TEST_F(ModelPreferThreadsIntegrationTest, Direct_Apple_SpecialLatencyAndThroughput) {
    Config config;
    std::vector<std::vector<int>> proc_type_table = {{10, 6, 4, 0, 0, 0, 0}};
    ov::MemBandwidthPressure tolerance;
    float isaThreshold = 1.0f;
    configure_apple_threads(config, proc_type_table, tolerance, isaThreshold, false, false);
    // MAIN > EFFICIENT -> latency equals main
    EXPECT_EQ(config.modelPreferThreadsLatency, 6);
}

#endif

}  // namespace