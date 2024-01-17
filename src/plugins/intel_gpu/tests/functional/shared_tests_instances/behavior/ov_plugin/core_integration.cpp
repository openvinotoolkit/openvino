// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/core_integration.hpp"

#include "behavior/ov_plugin/core_integration_sw.hpp"
#include "behavior/ov_plugin/query_model.hpp"

using namespace ov::test::behavior;

namespace {

INSTANTIATE_TEST_SUITE_P(smoke_OVClassBasicTestP, OVClassBasicTestP,
        ::testing::Values(std::make_pair(std::string("openvino_intel_gpu_plugin"), std::string(ov::test::utils::DEVICE_GPU))));

INSTANTIATE_TEST_SUITE_P(smoke_OVClassNetworkTestP, OVClassNetworkTestP,
        ::testing::Values(std::string(ov::test::utils::DEVICE_GPU)));

//
// OV Class GetMetric
//

INSTANTIATE_TEST_SUITE_P(
        nightly_OVClassGetMetricTest, OVClassGetMetricTest_SUPPORTED_CONFIG_KEYS,
        ::testing::Values(std::string(ov::test::utils::DEVICE_GPU),
                          std::string(ov::test::utils::DEVICE_HETERO),
                          std::string(ov::test::utils::DEVICE_BATCH))
);

INSTANTIATE_TEST_SUITE_P(
        nightly_OVClassGetMetricTest, OVClassGetMetricTest_SUPPORTED_METRICS,
        ::testing::Values(std::string(ov::test::utils::DEVICE_GPU),
                          std::string(ov::test::utils::DEVICE_HETERO),
                          std::string(ov::test::utils::DEVICE_BATCH))
);

INSTANTIATE_TEST_SUITE_P(
        nightly_OVClassGetMetricTest, OVClassGetMetricTest_AVAILABLE_DEVICES,
        ::testing::Values(std::string(ov::test::utils::DEVICE_GPU))
);

INSTANTIATE_TEST_SUITE_P(
        nightly_OVClassGetMetricTest, OVClassGetMetricTest_FULL_DEVICE_NAME,
        ::testing::Values(std::string(ov::test::utils::DEVICE_GPU),
                          std::string(ov::test::utils::DEVICE_HETERO),
                          std::string(ov::test::utils::DEVICE_BATCH))
);

INSTANTIATE_TEST_SUITE_P(
        nightly_OVClassGetMetricTest, OVClassGetMetricTest_OPTIMIZATION_CAPABILITIES,
        ::testing::Values(std::string(ov::test::utils::DEVICE_GPU))
);

INSTANTIATE_TEST_SUITE_P(
        nightly_OVClassGetMetricTest, OVClassGetMetricTest_DEVICE_GOPS,
        ::testing::Values(std::string(ov::test::utils::DEVICE_GPU))
);

INSTANTIATE_TEST_SUITE_P(
        nightly_OVClassGetMetricTest, OVClassGetMetricTest_DEVICE_TYPE,
        ::testing::Values(std::string(ov::test::utils::DEVICE_GPU))
);

INSTANTIATE_TEST_SUITE_P(
        nightly_OVClassGetMetricTest, OVClassGetMetricTest_RANGE_FOR_ASYNC_INFER_REQUESTS,
        ::testing::Values(std::string(ov::test::utils::DEVICE_GPU))
);

INSTANTIATE_TEST_SUITE_P(
        nightly_OVClassGetMetricTest, OVClassGetMetricTest_RANGE_FOR_STREAMS,
        ::testing::Values(std::string(ov::test::utils::DEVICE_GPU))
);

INSTANTIATE_TEST_SUITE_P(
        nightly_OVClassGetMetricTest, OVClassGetMetricTest_ThrowUnsupported,
        ::testing::Values(std::string(ov::test::utils::DEVICE_GPU),
                          std::string(ov::test::utils::DEVICE_HETERO),
                          std::string(ov::test::utils::DEVICE_BATCH))
);

INSTANTIATE_TEST_SUITE_P(
        nightly_OVClassGetConfigTest, OVClassGetConfigTest_ThrowUnsupported,
        ::testing::Values(std::string(ov::test::utils::DEVICE_GPU),
                          std::string(ov::test::utils::DEVICE_HETERO),
                          std::string(ov::test::utils::DEVICE_BATCH))
);

INSTANTIATE_TEST_SUITE_P(
        nightly_OVClassGetAvailableDevices, OVClassGetAvailableDevices,
        ::testing::Values(std::string(ov::test::utils::DEVICE_GPU))
);

// IE Class Common tests with <pluginName, target_device params>
//
INSTANTIATE_TEST_SUITE_P(nightly_OVClassModelTestP, OVClassModelTestP, ::testing::Values("GPU"));
INSTANTIATE_TEST_SUITE_P(nightly_OVClassModelOptionalTestP, OVClassModelOptionalTestP, ::testing::Values("GPU"));

// Several devices case
INSTANTIATE_TEST_SUITE_P(nightly_OVClassSeveralDevicesTest,
                         OVClassSeveralDevicesTestCompileModel,
                         ::testing::Values(std::vector<std::string>({"GPU.0", "GPU.1"})));

INSTANTIATE_TEST_SUITE_P(nightly_OVClassSeveralDevicesTest,
                         OVClassSeveralDevicesTestQueryModel,
                         ::testing::Values(std::vector<std::string>({"GPU.0", "GPU.1"})));

INSTANTIATE_TEST_SUITE_P(nightly_OVClassSeveralDevicesTest,
                         OVClassSeveralDevicesTestDefaultCore,
                         ::testing::Values(std::vector<std::string>({"GPU.0", "GPU.1"})));

// Set config for all GPU devices

INSTANTIATE_TEST_SUITE_P(nightly_OVClassSetGlobalConfigTest, OVClassSetGlobalConfigTest, ::testing::Values("GPU"));

// IE Class Query network

INSTANTIATE_TEST_SUITE_P(smoke_OVClassQueryModelTest, OVClassQueryModelTest, ::testing::Values("GPU"));

}  // namespace
