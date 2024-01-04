// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/core_integration.hpp"

#include "behavior/ov_plugin/core_integration_sw.hpp"
#include "behavior/ov_plugin/query_model.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/properties.hpp"

using namespace ov::test::behavior;

// defined in plugin_name.cpp
extern const char* cpu_plugin_file_name;

namespace {
//
// OV Class Common tests with <pluginName, deviceName params>
//

INSTANTIATE_TEST_SUITE_P(smoke_OVClassCommon, OVClassBaseTestP, ::testing::Values("CPU"));

INSTANTIATE_TEST_SUITE_P(smoke_OVClassNetworkTestP, OVClassNetworkTestP, ::testing::Values("CPU"));

//
// OV Class GetMetric
//

INSTANTIATE_TEST_SUITE_P(smoke_OVClassGetMetricTest,
                         OVClassGetMetricTest_SUPPORTED_CONFIG_KEYS,
                         ::testing::Values("CPU", "HETERO"));

INSTANTIATE_TEST_SUITE_P(smoke_OVClassGetMetricTest,
                         OVClassGetMetricTest_SUPPORTED_METRICS,
                         ::testing::Values("CPU", "HETERO"));

INSTANTIATE_TEST_SUITE_P(smoke_OVClassGetMetricTest, OVClassGetMetricTest_AVAILABLE_DEVICES, ::testing::Values("CPU"));

INSTANTIATE_TEST_SUITE_P(smoke_OVClassGetMetricTest,
                         OVClassGetMetricTest_FULL_DEVICE_NAME,
                         ::testing::Values("CPU", "HETERO"));

INSTANTIATE_TEST_SUITE_P(smoke_OVClassGetMetricTest,
                         OVClassGetMetricTest_OPTIMIZATION_CAPABILITIES,
                         ::testing::Values("CPU"));

INSTANTIATE_TEST_SUITE_P(smoke_OVClassGetMetricTest,
                         OVClassGetMetricTest_RANGE_FOR_ASYNC_INFER_REQUESTS,
                         ::testing::Values("CPU"));

INSTANTIATE_TEST_SUITE_P(smoke_OVClassGetMetricTest, OVClassGetMetricTest_RANGE_FOR_STREAMS, ::testing::Values("CPU"));

INSTANTIATE_TEST_SUITE_P(smoke_OVClassGetMetricTest,
                         OVClassGetMetricTest_ThrowUnsupported,
                         ::testing::Values("CPU", "HETERO"));

INSTANTIATE_TEST_SUITE_P(smoke_OVClassGetConfigTest,
                         OVClassGetConfigTest_ThrowUnsupported,
                         ::testing::Values("CPU", "HETERO"));

INSTANTIATE_TEST_SUITE_P(smoke_OVClassGetAvailableDevices, OVClassGetAvailableDevices, ::testing::Values("CPU"));

//
// OV Class GetConfig
//

INSTANTIATE_TEST_SUITE_P(smoke_OVClassGetConfigTest, OVClassGetConfigTest, ::testing::Values("CPU"));

//////////////////////////////////////////////////////////////////////////////////////////

TEST(OVClassBasicTest, smoke_SetConfigAfterCreatedThrow) {
    ov::Core core;
    std::string value = {};

    ASSERT_NO_THROW(core.set_property("CPU", ov::inference_num_threads(1)));
    ASSERT_NO_THROW(value = core.get_property("CPU", ov::inference_num_threads.name()).as<std::string>());
    ASSERT_EQ("1", value);

    ASSERT_NO_THROW(core.set_property("CPU", ov::inference_num_threads(4)));
    ASSERT_NO_THROW(value = core.get_property("CPU", ov::inference_num_threads.name()).as<std::string>());
    ASSERT_EQ("4", value);
}

INSTANTIATE_TEST_SUITE_P(smoke_OVClassImportExportTestP, OVClassImportExportTestP, ::testing::Values("HETERO:CPU"));

// IE Class Query model
INSTANTIATE_TEST_SUITE_P(smoke_OVClassQueryModelTest, OVClassQueryModelTest, ::testing::Values("CPU"));

// OV Class Load network
INSTANTIATE_TEST_SUITE_P(smoke_OVClassLoadNetworkTest, OVClassLoadNetworkTest, ::testing::Values("CPU"));

INSTANTIATE_TEST_SUITE_P(smoke_OVClassLoadNetworkTest, OVClassLoadNetworkTestWithThrow, ::testing::Values(""));

}  // namespace
