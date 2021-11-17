// Copyright (C) 2018-2021 Intel Corporation
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
// IE Class Common tests with <pluginName, deviceName params>
//

INSTANTIATE_TEST_SUITE_P(
        nightly_IEClassCommon, IEClassBasicTestP,
        ::testing::Values(std::make_pair("clDNNPlugin", "GPU"))
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
        ::testing::Values("GPU", "MULTI", "HETERO", "AUTO")
);

INSTANTIATE_TEST_SUITE_P(
        nightly_IEClassGetMetricTest, IEClassGetMetricTest_SUPPORTED_METRICS,
        ::testing::Values("GPU", "MULTI", "HETERO", "AUTO")
);

INSTANTIATE_TEST_SUITE_P(
        nightly_IEClassGetMetricTest, IEClassGetMetricTest_AVAILABLE_DEVICES,
        ::testing::Values("GPU")
);

INSTANTIATE_TEST_SUITE_P(
        nightly_IEClassGetMetricTest, IEClassGetMetricTest_FULL_DEVICE_NAME,
        ::testing::Values("GPU", "MULTI", "HETERO", "AUTO")
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
        ::testing::Values("GPU", "MULTI", "HETERO", "AUTO")
);

INSTANTIATE_TEST_SUITE_P(
        nightly_IEClassGetConfigTest, IEClassGetConfigTest_ThrowUnsupported,
        ::testing::Values("GPU", "MULTI", "HETERO", "AUTO")
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

    ASSERT_NO_THROW(p = ie.GetMetric(deviceName, GPU_METRIC_KEY(DEVICE_TOTAL_MEM_SIZE)));
    uint64_t t = p;

    std::cout << "GPU device total memory size: " << t << std::endl;

    ASSERT_METRIC_SUPPORTED_IE(GPU_METRIC_KEY(DEVICE_TOTAL_MEM_SIZE));
}

INSTANTIATE_TEST_SUITE_P(
        nightly_IEClassGetMetricTest, IEClassGetMetricTest_GPU_DEVICE_TOTAL_MEM_SIZE,
        ::testing::Values("GPU")
);

using IEClassGetMetricTest_GPU_UARCH_VERSION = BehaviorTestsUtils::IEClassBaseTestP;
TEST_P(IEClassGetMetricTest_GPU_UARCH_VERSION, GetMetricAndPrintNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    InferenceEngine::Core ie;
    InferenceEngine::Parameter p;

    ASSERT_NO_THROW(p = ie.GetMetric(deviceName, GPU_METRIC_KEY(UARCH_VERSION)));
    std::string t = p;

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

    ASSERT_NO_THROW(p = ie.GetMetric(deviceName, GPU_METRIC_KEY(EXECUTION_UNITS_COUNT)));
    int t = p;

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

    InferenceEngine::ExecutableNetwork exec_net = ie.LoadNetwork(simpleCnnNetwork, deviceName);

    ASSERT_NO_THROW(p = ie.GetMetric(deviceName, GPU_METRIC_KEY(MEMORY_STATISTICS)));
    std::map<std::string, uint64_t> t = p;

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

    InferenceEngine::ExecutableNetwork exec_net1 = ie.LoadNetwork(simpleCnnNetwork, deviceName);

    ASSERT_NO_THROW(p = ie.GetMetric(deviceName, GPU_METRIC_KEY(MEMORY_STATISTICS)));
    std::map<std::string, uint64_t> t1 = p;

    ASSERT_FALSE(t1.empty());
    std::cout << "Memory Statistics after the 1st network is loaded: " << std::endl;
    for (auto &&kv : t1) {
        ASSERT_NE(kv.second, 0);
        std::cout << kv.first << ": " << kv.second << " bytes" << std::endl;
    }

    InferenceEngine::ExecutableNetwork exec_net2 = ie.LoadNetwork(simpleCnnNetwork, deviceName);

    ASSERT_NO_THROW(p = ie.GetMetric(deviceName, GPU_METRIC_KEY(MEMORY_STATISTICS)));
    std::map<std::string, uint64_t> t2 = p;

    ASSERT_FALSE(t2.empty());
    std::cout << "Memory Statistics after the 2nd network is loaded: " << std::endl;
    for (auto &&kv : t2) {
        ASSERT_NE(kv.second, 0);
        auto iter = t1.find(kv.first);
        if (iter != t1.end()) {
            ASSERT_EQ(kv.second, t1[kv.first] * 2);
        }
        std::cout << kv.first << ": " << kv.second << " bytes" << std::endl;
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

    ASSERT_NO_THROW(p = ie.GetMetric(deviceName, GPU_METRIC_KEY(MEMORY_STATISTICS)));
    std::map<std::string, uint64_t> t1 = p;
    ASSERT_TRUE(t1.empty());

    {
        InferenceEngine::ExecutableNetwork exec_net1 = ie.LoadNetwork(simpleCnnNetwork, deviceName);

        ASSERT_NO_THROW(p = ie.GetMetric(deviceName, GPU_METRIC_KEY(MEMORY_STATISTICS)));
        std::map<std::string, uint64_t> t2 = p;

        ASSERT_FALSE(t2.empty());
        std::cout << "Memory Statistics after the 1st network is loaded: " << std::endl;
        for (auto &&kv : t2) {
            ASSERT_NE(kv.second, 0);
            std::cout << kv.first << ": " << kv.second << " bytes" << std::endl;
        }
        {
            InferenceEngine::ExecutableNetwork exec_net2 = ie.LoadNetwork(actualCnnNetwork, deviceName);

            ASSERT_NO_THROW(p = ie.GetMetric(deviceName, GPU_METRIC_KEY(MEMORY_STATISTICS)));
            std::map<std::string, uint64_t> t3 = p;

            ASSERT_FALSE(t3.empty());
            std::cout << "Memory Statistics after the 2nd network is loaded: " << std::endl;
            for (auto &&kv : t3) {
                ASSERT_NE(kv.second, 0);
                std::cout << kv.first << ": " << kv.second << " bytes" << std::endl;
            }
        }
        ASSERT_NO_THROW(p = ie.GetMetric(deviceName, GPU_METRIC_KEY(MEMORY_STATISTICS)));
        std::map<std::string, uint64_t> t4 = p;

        ASSERT_FALSE(t4.empty());
        std::cout << "Memory Statistics after the 2nd LoadNetwork is released: " << std::endl;
        for (auto &&kv : t4) {
            ASSERT_NE(kv.second, 0);
            if (kv.first.find("_cur") != std::string::npos) {
                auto iter = t2.find(kv.first);
                if (iter != t2.end()) {
                    ASSERT_EQ(t2[kv.first], kv.second);
                }
            }
            std::cout << kv.first << ": " << kv.second << " bytes" << std::endl;
        }
    }
    ASSERT_NO_THROW(p = ie.GetMetric(deviceName, GPU_METRIC_KEY(MEMORY_STATISTICS)));
    std::map<std::string, uint64_t> t5 = p;

    ASSERT_FALSE(t5.empty());
    std::cout << "Memory Statistics after the 1st LoadNetwork is released: " << std::endl;
    for (auto &&kv : t5) {
        if (kv.first.find("_cur") != std::string::npos) {
            ASSERT_EQ(kv.second, 0);
        }
        std::cout << kv.first << ": " << kv.second << " bytes" << std::endl;
    }
    ASSERT_METRIC_SUPPORTED_IE(GPU_METRIC_KEY(MEMORY_STATISTICS));
}

INSTANTIATE_TEST_SUITE_P(
        nightly_IEClassGetMetricTest, IEClassGetMetricTest_GPU_MEMORY_STATISTICS_CHECK_VALUES,
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
