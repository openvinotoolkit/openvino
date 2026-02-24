#include <gtest/gtest.h>

#include <memory>

#include "common_test_utils/test_common.hpp"
#include "cpu_streams_calculation.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/runtime/performance_heuristics.hpp"
#include "openvino/runtime/threading/cpu_streams_info.hpp"

using namespace testing;
using namespace ov;
using namespace ov::intel_cpu;

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
    int num_sockets = proc_type_table.size() == 1 ? 1 : proc_type_table.size() - 1;
    float isaSpecificThreshold = 2.0f;
    int result =
        get_model_prefer_threads(num_streams, proc_type_table, model, config, num_sockets, isaSpecificThreshold);
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
    int num_sockets = proc_type_table.size() == 1 ? 1 : proc_type_table.size() - 1;
    float isaSpecificThreshold = 2.0f;
    int result =
        get_model_prefer_threads(num_streams, proc_type_table, model, config, num_sockets, isaSpecificThreshold);
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
    int num_sockets = proc_type_table.size() == 1 ? 1 : proc_type_table.size() - 1;
    float isaSpecificThreshold = 2.0f;
    int result =
        get_model_prefer_threads(num_streams, proc_type_table, model, config, num_sockets, isaSpecificThreshold);
    EXPECT_EQ(config.modelPreferThreadsLatency, 0);
    EXPECT_EQ(result, 0);
}

TEST_F(ModelPreferThreadsIntegrationTest, UnknownMemToleranceEdgeCase) {
    std::vector<std::vector<int>> proc_type_table = {{8, 8, 0, 0, 0, 0, 0}};
    auto model = make_dummy_model();
    Config config;
    config.modelType = Config::ModelType::CNN;
    config.modelPreferThreads = -1;
    int num_streams = 0;
    int num_sockets = proc_type_table.size() == 1 ? 1 : proc_type_table.size() - 1;
    float isaSpecificThreshold = 2.0f;
    int result =
        get_model_prefer_threads(num_streams, proc_type_table, model, config, num_sockets, isaSpecificThreshold);
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
    int result = get_model_prefer_threads(num_streams, proc_type_table, model, config, 1, 1.0f);
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
    // num_streams <= num_sockets -> latency path
    int num_sockets = proc_type_table.size() == 1 ? 1 : proc_type_table.size() - 1;
    float isaSpecificThreshold = 2.0f;
    int result = get_model_prefer_threads(1, proc_type_table, model, config, num_sockets, isaSpecificThreshold);
    EXPECT_EQ(result, config.modelPreferThreadsLatency);

    // num_streams > num_sockets -> throughput path
    result = get_model_prefer_threads(2, proc_type_table, model, config, num_sockets, isaSpecificThreshold);
    EXPECT_EQ(result, config.modelPreferThreadsThroughput);

    // num_streams == 0 -> throughput path regardless of num_sockets
    result = get_model_prefer_threads(0, proc_type_table, model, config, num_sockets, isaSpecificThreshold);
    EXPECT_EQ(result, config.modelPreferThreadsThroughput);
}

TEST_F(ModelPreferThreadsIntegrationTest, IsaSpecificThreshold_DefaultAndCustomCoverage) {
    std::vector<std::vector<int>> proc_type_table = {{8, 8, 0, 0, 0, 0, 0}};
    auto model = make_dummy_model();

    Config default_isa_config;
    default_isa_config.modelType = Config::ModelType::CNN;
    default_isa_config.modelPreferThreads = -1;
    int result_default = get_model_prefer_threads(0, proc_type_table, model, default_isa_config, 1, -1.0f);
    EXPECT_EQ(result_default, default_isa_config.modelPreferThreadsThroughput);

    Config custom_isa_config_low;
    custom_isa_config_low.modelType = Config::ModelType::CNN;
    custom_isa_config_low.modelPreferThreads = -1;
    int result_isa_low = get_model_prefer_threads(0, proc_type_table, model, custom_isa_config_low, 1, 1.0f);
    EXPECT_EQ(result_isa_low, custom_isa_config_low.modelPreferThreadsThroughput);

    Config custom_isa_config_high;
    custom_isa_config_high.modelType = Config::ModelType::CNN;
    custom_isa_config_high.modelPreferThreads = -1;
    int result_isa_high = get_model_prefer_threads(0, proc_type_table, model, custom_isa_config_high, 1, 2.0f);
    EXPECT_EQ(result_isa_high, custom_isa_config_high.modelPreferThreadsThroughput);

    // Higher ISA threshold factor should not increase preferred throughput threads.
    EXPECT_LE(custom_isa_config_high.modelPreferThreadsThroughput, custom_isa_config_low.modelPreferThreadsThroughput);
}

TEST_F(ModelPreferThreadsIntegrationTest, NumSockets_DefaultAndCustomCoverage) {
    std::vector<std::vector<int>> proc_type_table = {{8, 8, 0, 0, 0, 0, 0}};
    auto model = make_dummy_model();

    Config default_socket_config;
    default_socket_config.modelType = Config::ModelType::CNN;
    default_socket_config.modelPreferThreads = -1;
    int num_sockets = proc_type_table.size() == 1 ? 1 : proc_type_table.size() - 1;
    float isaSpecificThreshold = 2.0f;
    int result_default_socket =
        get_model_prefer_threads(1, proc_type_table, model, default_socket_config, num_sockets, isaSpecificThreshold);
    EXPECT_EQ(result_default_socket, default_socket_config.modelPreferThreadsLatency);

    Config custom_socket_config;
    custom_socket_config.modelType = Config::ModelType::CNN;
    custom_socket_config.modelPreferThreads = -1;
    int result_custom_socket =
        get_model_prefer_threads(2, proc_type_table, model, custom_socket_config, num_sockets, isaSpecificThreshold);
    EXPECT_EQ(result_custom_socket, custom_socket_config.modelPreferThreadsThroughput);
}

#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)

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
    config.threads = 0;
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

    configure_x86_throughput_threads(config, proc_type_table, tolerance, 1.0f);
    EXPECT_GE(config.modelPreferThreadsThroughput, 1);
}

TEST_F(ModelPreferThreadsIntegrationTest, Direct_X86_Hybrid_LP_AutoCase) {
    Config config;
    std::vector<std::vector<int>> proc_type_table = {{12, 4, 0, 8, 0, 0, 0}};
    ov::MemBandwidthPressure tolerance;
    tolerance.max_mem_tolerance = 0.08f;
    tolerance.total_light_convs = 11;

    configure_x86_hybrid_lp_threads(config, proc_type_table, tolerance);
    EXPECT_EQ(config.modelPreferThreadsLatency, 12);
}

TEST_F(ModelPreferThreadsIntegrationTest, X86_NonHybrid_ZeroMainUsesEfficient) {
    std::vector<std::vector<int>> proc_type_table = {{4, 0, 4, 0, 0, 0, 0}};
    auto model = make_dummy_model();
    Config config;
    config.modelType = Config::ModelType::CNN;
    config.modelPreferThreads = -1;
    int num_streams = 1;
    int num_sockets = proc_type_table.size() == 1 ? 1 : proc_type_table.size() - 1;
    float isaSpecificThreshold = 2.0f;
    int result =
        get_model_prefer_threads(num_streams, proc_type_table, model, config, num_sockets, isaSpecificThreshold);
    EXPECT_EQ(config.modelPreferThreadsLatency, 4);
    EXPECT_EQ(result, 4);
}

TEST_F(ModelPreferThreadsIntegrationTest, X86_Hybrid_HybridApplicable_IsLLM_MainOnly) {
    std::vector<std::vector<int>> proc_type_table = {{16, 4, 8, 4, 0, 0, 0}};
    auto model = make_dummy_model();
    Config config;
    config.modelType = Config::ModelType::LLM;
    config.modelPreferThreads = -1;
    int num_streams = 1;
    int num_sockets = proc_type_table.size() == 1 ? 1 : proc_type_table.size() - 1;
    float isaSpecificThreshold = 2.0f;
    int result =
        get_model_prefer_threads(num_streams, proc_type_table, model, config, num_sockets, isaSpecificThreshold);
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
    int num_sockets = proc_type_table.size() == 1 ? 1 : proc_type_table.size() - 1;
    float isaSpecificThreshold = 2.0f;
    int result =
        get_model_prefer_threads(num_streams, proc_type_table, model, config, num_sockets, isaSpecificThreshold);
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
    int num_sockets = proc_type_table.size() == 1 ? 1 : proc_type_table.size() - 1;
    float isaSpecificThreshold = 2.0f;
    int result =
        get_model_prefer_threads(num_streams, proc_type_table, model, config, num_sockets, isaSpecificThreshold);
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
    int num_sockets = proc_type_table.size() == 1 ? 1 : proc_type_table.size() - 1;
    float isaSpecificThreshold = 2.0f;
    int result =
        get_model_prefer_threads(num_streams, proc_type_table, model, config, num_sockets, isaSpecificThreshold);
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
    int num_sockets = proc_type_table.size() == 1 ? 1 : proc_type_table.size() - 1;
    float isaSpecificThreshold = 2.0f;
    int result =
        get_model_prefer_threads(num_streams, proc_type_table, model, config, num_sockets, isaSpecificThreshold);
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
    int num_sockets = proc_type_table.size() == 1 ? 1 : proc_type_table.size() - 1;
    float isaSpecificThreshold = 2.0f;
    int result =
        get_model_prefer_threads(num_streams, proc_type_table, model, config, num_sockets, isaSpecificThreshold);
    EXPECT_EQ(config.modelPreferThreadsLatency, 4);
    EXPECT_EQ(result, config.modelPreferThreadsLatency);
}

#endif

#if defined(OPENVINO_ARCH_ARM) && defined(__linux__)
TEST_F(ModelPreferThreadsIntegrationTest, ARM_Linux_Throughput_UnknownAndMemLimited) {
    std::vector<std::vector<int>> proc_type_table = {{8, 4, 4, 0, 0, 0, 0}};
    auto model = make_dummy_model();
    Config config;
    config.modelType = Config::ModelType::CNN;
    config.modelPreferThreads = -1;

    int result = get_model_prefer_threads(0, proc_type_table, model, config, 1, 2.0f);
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
    int result = get_model_prefer_threads(1, proc_type_table, model, config, 1, 1.0f);
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
    int result = get_model_prefer_threads(num_streams, proc_type_table, model, config, 1, 1.0f);
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