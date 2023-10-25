// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/plugin/core_integration.hpp"

#ifdef _WIN32
# include "gpu/gpu_context_api_dx.hpp"
#elif defined ENABLE_LIBVA
# include <gpu/gpu_context_api_va.hpp>
#endif
#include "gpu/gpu_context_api_ocl.hpp"

#include "gpu/gpu_config.hpp"

using namespace BehaviorTestsDefinitions;

namespace {
// IE Class Common tests with <pluginName, target_device params>
//

INSTANTIATE_TEST_SUITE_P(
        nightly_IEClassCommon, IEClassBasicTestP,
        ::testing::Values(std::make_pair("openvino_intel_gpu_plugin", "GPU"))
);

INSTANTIATE_TEST_SUITE_P(
        nightly_IEClassNetworkTestP, IEClassNetworkTestP,
        ::testing::Values("GPU")
);

//
// IE Class GetMetric
//

INSTANTIATE_TEST_SUITE_P(
        nightly_IEClassGetMetricTest, IEClassGetMetricTest_SUPPORTED_CONFIG_KEYS,
        ::testing::Values("GPU", "HETERO", "BATCH")
);

INSTANTIATE_TEST_SUITE_P(
        nightly_IEClassGetMetricTest, IEClassGetMetricTest_SUPPORTED_METRICS,
        ::testing::Values("GPU", "HETERO", "BATCH")
);

INSTANTIATE_TEST_SUITE_P(
        nightly_IEClassGetMetricTest, IEClassGetMetricTest_AVAILABLE_DEVICES,
        ::testing::Values("GPU")
);

INSTANTIATE_TEST_SUITE_P(
        nightly_IEClassGetMetricTest, IEClassGetMetricTest_FULL_DEVICE_NAME,
        ::testing::Values("GPU", "HETERO", "BATCH")
);

INSTANTIATE_TEST_SUITE_P(
        nightly_IEClassGetMetricTest, IEClassGetMetricTest_OPTIMIZATION_CAPABILITIES,
        ::testing::Values("GPU")
);

INSTANTIATE_TEST_SUITE_P(
        nightly_IEClassGetMetricTest, IEClassGetMetricTest_DEVICE_GOPS,
        ::testing::Values("GPU")
);

INSTANTIATE_TEST_SUITE_P(
        nightly_IEClassGetMetricTest, IEClassGetMetricTest_DEVICE_TYPE,
        ::testing::Values("GPU")
);

INSTANTIATE_TEST_SUITE_P(
        nightly_IEClassGetMetricTest, IEClassGetMetricTest_RANGE_FOR_ASYNC_INFER_REQUESTS,
        ::testing::Values("GPU")
);

INSTANTIATE_TEST_SUITE_P(
        nightly_IEClassGetMetricTest, IEClassGetMetricTest_RANGE_FOR_STREAMS,
        ::testing::Values("GPU")
);

INSTANTIATE_TEST_SUITE_P(
        nightly_IEClassGetMetricTest, IEClassGetMetricTest_ThrowUnsupported,
        ::testing::Values("GPU", "HETERO", "BATCH")
);

INSTANTIATE_TEST_SUITE_P(
        nightly_IEClassGetConfigTest, IEClassGetConfigTest_ThrowUnsupported,
        ::testing::Values("GPU", "HETERO", "BATCH")
);

INSTANTIATE_TEST_SUITE_P(
        nightly_IEClassGetAvailableDevices, IEClassGetAvailableDevices,
        ::testing::Values("GPU")
);

//
// GPU specific metrics
//
using IEClassGetMetricTest_GPU_DEVICE_TOTAL_MEM_SIZE = BehaviorTestsUtils::IEClassBaseTestP;
TEST_P(IEClassGetMetricTest_GPU_DEVICE_TOTAL_MEM_SIZE, GetMetricAndPrintNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    InferenceEngine::Core ie;
    InferenceEngine::Parameter p;

    ASSERT_NO_THROW(p = ie.GetMetric(target_device, GPU_METRIC_KEY(DEVICE_TOTAL_MEM_SIZE)));
    auto t = p.as<uint64_t>();

    std::cout << "GPU device total memory size: " << t << std::endl;

    ASSERT_METRIC_SUPPORTED_IE(GPU_METRIC_KEY(DEVICE_TOTAL_MEM_SIZE));
}

INSTANTIATE_TEST_SUITE_P(
        nightly_IEClassGetMetricTest, IEClassGetMetricTest_GPU_DEVICE_TOTAL_MEM_SIZE,
        ::testing::Values("GPU")
);

using IEClassGetMetricTest_GPU_OPTIMAL_BATCH_SIZE = BehaviorTestsUtils::IEClassBaseTestP;
TEST_P(IEClassGetMetricTest_GPU_OPTIMAL_BATCH_SIZE, GetMetricAndPrintNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    InferenceEngine::Core ie;
    InferenceEngine::Parameter p;

    std::map<std::string, InferenceEngine::Parameter> _options = {{"MODEL_PTR", simpleCnnNetwork.getFunction()}};
    ASSERT_NO_THROW(p = ie.GetMetric(target_device, METRIC_KEY(OPTIMAL_BATCH_SIZE), _options).as<unsigned int>());
    auto t = p.as<unsigned int>();

    std::cout << "GPU device optimal batch size: " << t << std::endl;

    ASSERT_METRIC_SUPPORTED_IE(METRIC_KEY(OPTIMAL_BATCH_SIZE));
}

INSTANTIATE_TEST_SUITE_P(
        nightly_IEClassExecutableNetworkGetMetricTest, IEClassGetMetricTest_GPU_OPTIMAL_BATCH_SIZE,
        ::testing::Values("GPU")
);

using IEClassGetMetricTest_GPU_MAX_BATCH_SIZE_DEFAULT = BehaviorTestsUtils::IEClassBaseTestP;
TEST_P(IEClassGetMetricTest_GPU_MAX_BATCH_SIZE_DEFAULT, GetMetricAndPrintNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    InferenceEngine::Core ie;
    InferenceEngine::Parameter p;

    std::map<std::string, InferenceEngine::Parameter> _options = {{"MODEL_PTR", simpleCnnNetwork.getFunction()}};
    ASSERT_NO_THROW(p = ie.GetMetric(target_device, METRIC_KEY(MAX_BATCH_SIZE), _options).as<uint32_t>());
    auto t = p.as<uint32_t>();

    std::cout << "GPU device max available batch size: " << t << std::endl;

    ASSERT_METRIC_SUPPORTED_IE(METRIC_KEY(MAX_BATCH_SIZE));
}

INSTANTIATE_TEST_SUITE_P(
        nightly_IEClassExecutableNetworkGetMetricTest, IEClassGetMetricTest_GPU_MAX_BATCH_SIZE_DEFAULT,
        ::testing::Values("GPU")
);

using IEClassGetMetricTest_GPU_MAX_BATCH_SIZE_STREAM_DEVICE_MEM = BehaviorTestsUtils::IEClassBaseTestP;
TEST_P(IEClassGetMetricTest_GPU_MAX_BATCH_SIZE_STREAM_DEVICE_MEM, GetMetricAndPrintNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    InferenceEngine::Core ie;
    InferenceEngine::Parameter p;
    uint32_t n_streams = 2;
    int64_t available_device_mem_size = 1073741824;
    std::map<std::string, InferenceEngine::Parameter> _options = {{"MODEL_PTR", simpleCnnNetwork.getFunction()}};
    _options.insert(std::make_pair("GPU_THROUGHPUT_STREAMS", n_streams));
    _options.insert(std::make_pair("AVAILABLE_DEVICE_MEM_SIZE", available_device_mem_size));

    ASSERT_NO_THROW(p = ie.GetMetric(target_device, METRIC_KEY(MAX_BATCH_SIZE), _options).as<uint32_t>());

    auto t = p.as<uint32_t>();

    std::cout << "GPU device max available batch size: " << t << std::endl;

    ASSERT_METRIC_SUPPORTED_IE(METRIC_KEY(MAX_BATCH_SIZE));
}

INSTANTIATE_TEST_SUITE_P(
        nightly_IEClassExecutableNetworkGetMetricTest, IEClassGetMetricTest_GPU_MAX_BATCH_SIZE_STREAM_DEVICE_MEM,
        ::testing::Values("GPU")
);

using IEClassGetMetricTest_GPU_UARCH_VERSION = BehaviorTestsUtils::IEClassBaseTestP;
TEST_P(IEClassGetMetricTest_GPU_UARCH_VERSION, GetMetricAndPrintNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    InferenceEngine::Core ie;
    InferenceEngine::Parameter p;

    ASSERT_NO_THROW(p = ie.GetMetric(target_device, GPU_METRIC_KEY(UARCH_VERSION)));
    auto t = p.as<std::string>();

    std::cout << "GPU device uarch: " << t << std::endl;

    ASSERT_METRIC_SUPPORTED_IE(GPU_METRIC_KEY(UARCH_VERSION));
}

INSTANTIATE_TEST_SUITE_P(
        nightly_IEClassGetMetricTest, IEClassGetMetricTest_GPU_UARCH_VERSION,
        ::testing::Values("GPU")
);

using IEClassGetMetricTest_GPU_EXECUTION_UNITS_COUNT = BehaviorTestsUtils::IEClassBaseTestP;
TEST_P(IEClassGetMetricTest_GPU_EXECUTION_UNITS_COUNT, GetMetricAndPrintNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    InferenceEngine::Core ie;
    InferenceEngine::Parameter p;

    ASSERT_NO_THROW(p = ie.GetMetric(target_device, GPU_METRIC_KEY(EXECUTION_UNITS_COUNT)));
    auto t = p.as<int>();

    std::cout << "GPU EUs count: " << t << std::endl;

    ASSERT_METRIC_SUPPORTED_IE(GPU_METRIC_KEY(EXECUTION_UNITS_COUNT));
}

INSTANTIATE_TEST_SUITE_P(
        nightly_IEClassGetMetricTest, IEClassGetMetricTest_GPU_EXECUTION_UNITS_COUNT,
        ::testing::Values("GPU")
);

using IEClassGetMetricTest_GPU_MEMORY_STATISTICS_DEFAULT = BehaviorTestsUtils::IEClassBaseTestP;
TEST_P(IEClassGetMetricTest_GPU_MEMORY_STATISTICS_DEFAULT, GetMetricAndPrintNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    InferenceEngine::Core ie;
    InferenceEngine::Parameter p;

    InferenceEngine::ExecutableNetwork exec_net = ie.LoadNetwork(simpleCnnNetwork, target_device);

    ASSERT_NO_THROW(p = ie.GetMetric(target_device, GPU_METRIC_KEY(MEMORY_STATISTICS)));
    auto t = p.as<std::map<std::string, uint64_t>>();

    ASSERT_FALSE(t.empty());
    std::cout << "Memory Statistics: " << std::endl;
    for (auto &&kv : t) {
        ASSERT_NE(kv.second, 0);
        std::cout << kv.first << ": " << kv.second << " bytes" << std::endl;
    }

    ASSERT_METRIC_SUPPORTED_IE(GPU_METRIC_KEY(MEMORY_STATISTICS));
}

INSTANTIATE_TEST_SUITE_P(
        nightly_IEClassGetMetricTest, IEClassGetMetricTest_GPU_MEMORY_STATISTICS_DEFAULT,
        ::testing::Values("GPU")
);

using IEClassGetMetricTest_GPU_MEMORY_STATISTICS_MULTIPLE_NETWORKS = BehaviorTestsUtils::IEClassBaseTestP;
TEST_P(IEClassGetMetricTest_GPU_MEMORY_STATISTICS_MULTIPLE_NETWORKS, GetMetricAndPrintNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    InferenceEngine::Core ie;
    InferenceEngine::Parameter p;

    InferenceEngine::ExecutableNetwork exec_net1 = ie.LoadNetwork(simpleCnnNetwork, target_device);

    ASSERT_NO_THROW(p = ie.GetMetric(target_device, GPU_METRIC_KEY(MEMORY_STATISTICS)));
    auto t1 = p.as<std::map<std::string, uint64_t>>();

    ASSERT_FALSE(t1.empty());
    for (auto &&kv : t1) {
        ASSERT_NE(kv.second, 0);
    }

    InferenceEngine::ExecutableNetwork exec_net2 = ie.LoadNetwork(simpleCnnNetwork, target_device);

    ASSERT_NO_THROW(p = ie.GetMetric(target_device, GPU_METRIC_KEY(MEMORY_STATISTICS)));
    auto t2 = p.as<std::map<std::string, uint64_t>>();

    ASSERT_FALSE(t2.empty());
    for (auto &&kv : t2) {
        ASSERT_NE(kv.second, 0);
        auto iter = t1.find(kv.first);
        if (iter != t1.end()) {
            ASSERT_EQ(kv.second, t1[kv.first] * 2);
        }
    }

    ASSERT_METRIC_SUPPORTED_IE(GPU_METRIC_KEY(MEMORY_STATISTICS));
}

INSTANTIATE_TEST_SUITE_P(
        nightly_IEClassGetMetricTest, IEClassGetMetricTest_GPU_MEMORY_STATISTICS_MULTIPLE_NETWORKS,
        ::testing::Values("GPU")
);

using IEClassGetMetricTest_GPU_MEMORY_STATISTICS_CHECK_VALUES = BehaviorTestsUtils::IEClassBaseTestP;
TEST_P(IEClassGetMetricTest_GPU_MEMORY_STATISTICS_CHECK_VALUES, GetMetricAndPrintNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    InferenceEngine::Core ie;
    InferenceEngine::Parameter p;

    ASSERT_NO_THROW(p = ie.GetMetric(target_device, GPU_METRIC_KEY(MEMORY_STATISTICS)));
    auto t1 = p.as<std::map<std::string, uint64_t>>();
    ASSERT_TRUE(t1.empty());

    {
        InferenceEngine::ExecutableNetwork exec_net1 = ie.LoadNetwork(simpleCnnNetwork, target_device);

        ASSERT_NO_THROW(p = ie.GetMetric(target_device, GPU_METRIC_KEY(MEMORY_STATISTICS)));
        auto t2 = p.as<std::map<std::string, uint64_t>>();

        ASSERT_FALSE(t2.empty());
        for (auto &&kv : t2) {
            ASSERT_NE(kv.second, 0);
        }
        {
            InferenceEngine::ExecutableNetwork exec_net2 = ie.LoadNetwork(actualCnnNetwork, target_device);

            ASSERT_NO_THROW(p = ie.GetMetric(target_device, GPU_METRIC_KEY(MEMORY_STATISTICS)));
            auto t3 = p.as<std::map<std::string, uint64_t>>();

            ASSERT_FALSE(t3.empty());
            for (auto &&kv : t3) {
                ASSERT_NE(kv.second, 0);
            }
        }
        ASSERT_NO_THROW(p = ie.GetMetric(target_device, GPU_METRIC_KEY(MEMORY_STATISTICS)));
        auto t4 = p.as<std::map<std::string, uint64_t>>();

        ASSERT_FALSE(t4.empty());
        for (auto &&kv : t4) {
            ASSERT_NE(kv.second, 0);
            if (kv.first.find("_cur") != std::string::npos) {
                auto iter = t2.find(kv.first);
                if (iter != t2.end()) {
                    ASSERT_EQ(t2[kv.first], kv.second);
                }
            }
        }
    }
    ASSERT_NO_THROW(p = ie.GetMetric(target_device, GPU_METRIC_KEY(MEMORY_STATISTICS)));
    auto t5 = p.as<std::map<std::string, uint64_t>>();

    ASSERT_FALSE(t5.empty());
    for (auto &&kv : t5) {
        if (kv.first.find("_cur") != std::string::npos) {
            ASSERT_EQ(kv.second, 0);
        }
    }
    ASSERT_METRIC_SUPPORTED_IE(GPU_METRIC_KEY(MEMORY_STATISTICS));
}

INSTANTIATE_TEST_SUITE_P(
        nightly_IEClassGetMetricTest, IEClassGetMetricTest_GPU_MEMORY_STATISTICS_CHECK_VALUES,
        ::testing::Values("GPU")
);

using IEClassGetMetricTest_GPU_MEMORY_STATISTICS_MULTI_THREADS = BehaviorTestsUtils::IEClassBaseTestP;
TEST_P(IEClassGetMetricTest_GPU_MEMORY_STATISTICS_MULTI_THREADS, GetMetricAndPrintNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    InferenceEngine::Core ie;
    InferenceEngine::Parameter p;

    std::atomic<uint32_t> counter{0u};
    std::vector<std::thread> threads(2);
    // key: thread id, value: executable network
    std::map<uint32_t, InferenceEngine::ExecutableNetwork> exec_net_map;
    std::vector<InferenceEngine::CNNNetwork> networks;
    networks.emplace_back(simpleCnnNetwork);
    networks.emplace_back(simpleCnnNetwork);

    InferenceEngine::ExecutableNetwork exec_net1 = ie.LoadNetwork(simpleCnnNetwork, target_device);

    ASSERT_NO_THROW(p = ie.GetMetric(target_device, GPU_METRIC_KEY(MEMORY_STATISTICS)));
    auto t1 = p.as<std::map<std::string, uint64_t>>();

    ASSERT_FALSE(t1.empty());
    for (auto &&kv : t1) {
        ASSERT_NE(kv.second, 0);
    }

    for (auto & thread : threads) {
        thread = std::thread([&](){
            auto value = counter++;
            exec_net_map[value] = ie.LoadNetwork(networks[value], target_device);
        });
    }

    for (auto & thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }

    ASSERT_NO_THROW(p = ie.GetMetric(target_device, GPU_METRIC_KEY(MEMORY_STATISTICS)));
    auto t2 = p.as<std::map<std::string, uint64_t>>();

    ASSERT_FALSE(t2.empty());
    for (auto &&kv : t2) {
        ASSERT_NE(kv.second, 0);
        auto iter = t1.find(kv.first);
        if (iter != t1.end()) {
            ASSERT_EQ(kv.second, t1[kv.first] * 3);
        }
    }

    ASSERT_METRIC_SUPPORTED_IE(GPU_METRIC_KEY(MEMORY_STATISTICS));
}

INSTANTIATE_TEST_SUITE_P(
        nightly_IEClassGetMetricTest, IEClassGetMetricTest_GPU_MEMORY_STATISTICS_MULTI_THREADS,
        ::testing::Values("GPU")
);

//
// IE Class GetConfig
//

INSTANTIATE_TEST_SUITE_P(
        nightly_IEClassGetConfigTest, IEClassGetConfigTest,
        ::testing::Values("GPU")
);

// IE Class Query network

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassQueryNetworkTest, IEClassQueryNetworkTest,
        ::testing::Values("GPU")
);

// IE Class Load network

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassLoadNetworkTest, IEClassLoadNetworkTest,
        ::testing::Values("GPU")
);

INSTANTIATE_TEST_SUITE_P(
        smoke_IEClassHeteroExecutableNetworkGetMetricTest, IEClassLoadNetworkAfterCoreRecreateTest,
        ::testing::Values("GPU")
);

// GetConfig / SetConfig for specific device

INSTANTIATE_TEST_SUITE_P(
        nightly_IEClassSpecificDevice0Test, IEClassSpecificDeviceTestGetConfig,
        ::testing::Values("GPU.0")
);

INSTANTIATE_TEST_SUITE_P(
        nightly_IEClassSpecificDevice1Test, IEClassSpecificDeviceTestGetConfig,
        ::testing::Values("GPU.1")
);

INSTANTIATE_TEST_SUITE_P(
        nightly_IEClassSpecificDevice0Test, IEClassSpecificDeviceTestSetConfig,
        ::testing::Values("GPU.0")
);

INSTANTIATE_TEST_SUITE_P(
        nightly_IEClassSpecificDevice1Test, IEClassSpecificDeviceTestSetConfig,
        ::testing::Values("GPU.1")
);

// Several devices case

INSTANTIATE_TEST_SUITE_P(
        nightly_IEClassSeveralDevicesTest, IEClassSeveralDevicesTestLoadNetwork,
        ::testing::Values(std::vector<std::string>({"GPU.0", "GPU.1"}))
);

INSTANTIATE_TEST_SUITE_P(
        nightly_IEClassSeveralDevicesTest, IEClassSeveralDevicesTestQueryNetwork,
        ::testing::Values(std::vector<std::string>({"GPU.0", "GPU.1"}))
);

INSTANTIATE_TEST_SUITE_P(
        nightly_IEClassSeveralDevicesTest, IEClassSeveralDevicesTestDefaultCore,
        ::testing::Values(std::vector<std::string>({"GPU.0", "GPU.1"}))
);

// Set default device ID

INSTANTIATE_TEST_SUITE_P(
        nightly_IEClassSetDefaultDeviceIDTest, IEClassSetDefaultDeviceIDTest,
        ::testing::Values(std::make_pair("GPU", "1"))
);

// Set config for all GPU devices

INSTANTIATE_TEST_SUITE_P(
        nightly_IEClassSetGlobalConfigTest, IEClassSetGlobalConfigTest,
        ::testing::Values("GPU")
);

} // namespace
