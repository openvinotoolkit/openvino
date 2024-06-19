// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "overload/ov_plugin/core_integration.hpp"
#include "behavior/compiled_model/properties.hpp"
#include "behavior/ov_plugin/core_integration_sw.hpp"
#include "behavior/ov_plugin/properties_tests.hpp"
#include "common/functions.h"
#include "common/utils.hpp"
#include "common/npu_test_env_cfg.hpp"
#include "common_test_utils/subgraph_builders/conv_pool_relu.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "functional_test_utils/ov_plugin_cache.hpp"
#include "intel_npu/al/config/common.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"

using namespace ov::test::behavior;

namespace {

const char* NPU_PLUGIN_LIB_NAME = "openvino_intel_npu_plugin";


std::vector<std::string> devices = {
        std::string(ov::test::utils::DEVICE_NPU),
};

std::pair<std::string, std::string> plugins[] = {
        std::make_pair(std::string(NPU_PLUGIN_LIB_NAME), std::string(ov::test::utils::DEVICE_NPU)),
};

namespace OVClassBasicTestName {
static std::string getTestCaseName(testing::TestParamInfo<std::pair<std::string, std::string>> obj) {
    std::ostringstream result;
    result << "OVClassBasicTestName_" << obj.param.first << "_" << obj.param.second;
    result << "_targetDevice=" << ov::test::utils::getTestsPlatformFromEnvironmentOr(ov::test::utils::DEVICE_NPU);

    return result.str();
}
}  // namespace OVClassBasicTestName

namespace OVClassNetworkTestName {
static std::string getTestCaseName(testing::TestParamInfo<std::string> obj) {
    std::ostringstream result;
    result << "OVClassNetworkTestName_" << obj.param;
    result << "_targetDevice=" << ov::test::utils::getTestsPlatformFromEnvironmentOr(ov::test::utils::DEVICE_NPU);

    return result.str();
}
}  // namespace OVClassNetworkTestName

//
// IE Class Common tests with <pluginName, deviceName params>
//

const std::vector<ov::AnyMap> configs = {{}};

INSTANTIATE_TEST_SUITE_P(backwardDrvComp_smoke_BehaviorTests_OVBasicPropertiesTestsP, OVBasicPropertiesTestsP,
                         ::testing::ValuesIn(plugins), OVClassBasicTestName::getTestCaseName);

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
INSTANTIATE_TEST_SUITE_P(backwardDrvComp_smoke_BehaviorTests_OVClassBasicTestP, OVClassBasicTestPNPU, ::testing::ValuesIn(plugins),
                         OVClassBasicTestName::getTestCaseName);
#endif

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests_OVClassNetworkTestP, OVClassNetworkTestPNPU,
                         ::testing::Combine(::testing::ValuesIn(devices), ::testing::ValuesIn(configs)),
                         OVClassNetworkTestPNPU::getTestCaseName);

//
// IE Class GetMetric
//

INSTANTIATE_TEST_SUITE_P(backwardDrvComp_smoke_BehaviorTests_OVGetMetricPropsTest_nightly, OVGetMetricPropsTest, ::testing::ValuesIn(devices),
                         OVClassNetworkTestName::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(backwardDrvComp_smoke_BehaviorTests_OVGetMetricPropsTest_nightly, OVGetMetricPropsOptionalTest,
                         ::testing::ValuesIn(devices), OVClassNetworkTestName::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        smoke_BehaviorTests_OVCheckSetSupportedRWMandatoryMetricsPropsTests, OVCheckSetSupportedRWMetricsPropsTests,
        ::testing::Combine(::testing::Values("MULTI", "AUTO"),
                           ::testing::ValuesIn(OVCheckSetSupportedRWMetricsPropsTests::getRWOptionalPropertiesValues(
                                   {ov::log::level.name()}))),
        ov::test::utils::appendPlatformTypeTestName<OVCheckSetSupportedRWMetricsPropsTests>);

const std::vector<ov::AnyMap> multiConfigs = {{ov::device::priorities(ov::test::utils::DEVICE_NPU)}};
const std::vector<ov::AnyMap> configsDeviceProperties = {
        {ov::device::properties(ov::test::utils::DEVICE_NPU, ov::num_streams(4))}};
const std::vector<ov::AnyMap> configsWithSecondaryProperties = {
        {ov::device::properties(ov::test::utils::DEVICE_NPU,
                                ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT))},
        {ov::device::properties(ov::test::utils::DEVICE_NPU,
                                ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)),
         ov::device::properties(ov::test::utils::DEVICE_NPU,
                                ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY))}};

const std::vector<ov::AnyMap> multiConfigsWithSecondaryProperties = {
        {ov::device::priorities(ov::test::utils::DEVICE_CPU),
         ov::device::properties(ov::test::utils::DEVICE_CPU,
                                ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT))},
        {ov::device::priorities(ov::test::utils::DEVICE_CPU),
         ov::device::properties(ov::test::utils::DEVICE_CPU,
                                ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)),
         ov::device::properties(ov::test::utils::DEVICE_NPU,
                                ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY))}};

const std::vector<ov::AnyMap> autoConfigsWithSecondaryProperties = {
        {ov::device::priorities(ov::test::utils::DEVICE_CPU),
         ov::device::properties("AUTO", ov::enable_profiling(false),
                                ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT))},
        {ov::device::priorities(ov::test::utils::DEVICE_CPU),
         ov::device::properties(ov::test::utils::DEVICE_CPU,
                                ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT))},
        {ov::device::priorities(ov::test::utils::DEVICE_CPU),
         ov::device::properties(ov::test::utils::DEVICE_CPU,
                                ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)),
         ov::device::properties(ov::test::utils::DEVICE_NPU,
                                ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY))},
        {ov::device::priorities(ov::test::utils::DEVICE_CPU),
         ov::device::properties("AUTO", ov::enable_profiling(false),
                                ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)),
         ov::device::properties(ov::test::utils::DEVICE_CPU,
                                ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT))},
        {ov::device::priorities(ov::test::utils::DEVICE_CPU),
         ov::device::properties("AUTO", ov::enable_profiling(false),
                                ov::device::priorities(ov::test::utils::DEVICE_NPU),
                                ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)),
         ov::device::properties(ov::test::utils::DEVICE_CPU,
                                ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)),
         ov::device::properties(ov::test::utils::DEVICE_NPU,
                                ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY))}};

//
// IE Class GetConfig
//

INSTANTIATE_TEST_SUITE_P(BehaviorTests_OVGetConfigTest_nightly, OVGetConfigTest, ::testing::ValuesIn(devices),
                         OVClassNetworkTestName::getTestCaseName);

// IE Class Load network

INSTANTIATE_TEST_SUITE_P(backwardDrvComp_smoke_BehaviorTests_OVClassLoadNetworkWithCorrectSecondaryPropertiesTest,
                         OVClassCompileModelWithCorrectPropertiesTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU, "AUTO:NPU", "MULTI:NPU"),
                                            ::testing::ValuesIn(configsWithSecondaryProperties)));

INSTANTIATE_TEST_SUITE_P(backwardDrvComp_smoke_Multi_BehaviorTests_OVClassCompileModelWithCorrectPropertiesTest,
                         OVClassCompileModelWithCorrectPropertiesTest,
                         ::testing::Combine(::testing::Values("MULTI"),
                                            ::testing::ValuesIn(multiConfigsWithSecondaryProperties)));

INSTANTIATE_TEST_SUITE_P(backwardDrvComp_smoke_AUTO_BehaviorTests_OVClassCompileModelWithCorrectPropertiesTest,
                         OVClassCompileModelWithCorrectPropertiesTest,
                         ::testing::Combine(::testing::Values("AUTO"),
                                            ::testing::ValuesIn(autoConfigsWithSecondaryProperties)));

// IE Class load and check network with ov::device::properties
// OVClassCompileModelAndCheckSecondaryPropertiesTest only works with property num_streams of type int32_t
INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_BehaviorTests_OVClassLoadNetworkAndCheckWithSecondaryPropertiesTest,
                         OVClassCompileModelAndCheckSecondaryPropertiesTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU, "AUTO:NPU", "MULTI:NPU"),
                                            ::testing::ValuesIn(configsDeviceProperties)));

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests_OVClassLoadNetworkTest, OVClassLoadNetworkTestNPU,
                         ::testing::Combine(::testing::ValuesIn(devices), ::testing::ValuesIn(configs)),
                         OVClassLoadNetworkTestNPU::getTestCaseName);

//
// NPU specific metrics
//

using OVClassGetMetricAndPrintNoThrow = OVClassBaseTestP;
TEST_P(OVClassGetMetricAndPrintNoThrow, DeviceAllocMemSizeLesserThanTotalMemSizeNPU) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::Core ie;
    ov::Any p;

    OV_ASSERT_NO_THROW(p = ie.get_property(target_device, ov::intel_npu::device_total_mem_size.name()));
    uint64_t t = p.as<uint64_t>();
    ASSERT_NE(t, 0);

    OV_ASSERT_NO_THROW(p = ie.get_property(target_device, ov::intel_npu::device_alloc_mem_size.name()));
    uint64_t a = p.as<uint64_t>();

    ASSERT_LT(a, t);

    std::cout << "OV NPU device alloc/total memory size: " << a << "/" << t << std::endl;
}

TEST_P(OVClassGetMetricAndPrintNoThrow, DeviceAllocMemSizeLesserAfterModelIsLoadedNPU) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::Core ie;
    ov::Any p;

    OV_ASSERT_NO_THROW(p = ie.get_property(target_device, ov::intel_npu::device_alloc_mem_size.name()));
    uint64_t a1 = p.as<uint64_t>();

    SKIP_IF_CURRENT_TEST_IS_DISABLED() {
        auto model = ov::test::utils::make_conv_pool_relu();
        OV_ASSERT_NO_THROW(ie.compile_model(
                model, target_device, {}));
    }

    OV_ASSERT_NO_THROW(p = ie.get_property(target_device, ov::intel_npu::device_alloc_mem_size.name()));
    uint64_t a2 = p.as<uint64_t>();

    std::cout << "OV NPU device {alloc before load network/alloc after load network} memory size: {" << a1 << "/" << a2
              << "}" << std::endl;

    // after the network is loaded onto device, allocated memory value should increase
    ASSERT_LE(a1, a2);
}

TEST_P(OVClassGetMetricAndPrintNoThrow, VpuDeviceAllocMemSizeLesserAfterModelIsLoaded) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::Core ie;
    ov::Any p;

    OV_ASSERT_NO_THROW(p = ie.get_property(target_device, ov::intel_npu::device_alloc_mem_size.name()));
    uint64_t a1 = p.as<uint64_t>();

    SKIP_IF_CURRENT_TEST_IS_DISABLED() {
        auto model = ov::test::utils::make_conv_pool_relu();
        OV_ASSERT_NO_THROW(
                ie.compile_model(model, target_device,
                                 ov::AnyMap{ov::log::level(ov::log::Level::DEBUG)}));
    }

    OV_ASSERT_NO_THROW(p = ie.get_property(target_device, ov::intel_npu::device_alloc_mem_size.name()));
    uint64_t a2 = p.as<uint64_t>();

    std::cout << "OV NPU device {alloc before load network/alloc after load network} memory size: {" << a1 << "/" << a2
              << "}" << std::endl;

    // after the network is loaded onto device, allocated memory value should increase
    ASSERT_LE(a1, a2);
}

TEST_P(OVClassGetMetricAndPrintNoThrow, DriverVersionNPU) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::Core ie;
    ov::Any p;

    OV_ASSERT_NO_THROW(p = ie.get_property(target_device, ov::intel_npu::driver_version.name()));
    uint32_t t = p.as<uint32_t>();

    std::cout << "NPU driver version is " << t << std::endl;

    OV_ASSERT_PROPERTY_SUPPORTED(ov::intel_npu::driver_version.name());
}

using OVClassCompileModel = OVClassBaseTestP;
TEST_P(OVClassCompileModel, CompileModelWithDifferentThreadNumbers) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::Core ie;
    ov::Any p;

    auto model = ov::test::utils::make_conv_pool_relu();
    OV_ASSERT_NO_THROW(ie.compile_model(model, target_device, {{ov::compilation_num_threads.name(), ov::Any(1)}}));

    OV_ASSERT_NO_THROW(ie.compile_model(model, target_device, {{ov::compilation_num_threads.name(), ov::Any(2)}}));

    OV_ASSERT_NO_THROW(ie.compile_model(model, target_device, {{ov::compilation_num_threads.name(), ov::Any(4)}}));

    EXPECT_ANY_THROW(ie.compile_model(model, target_device, {{ov::compilation_num_threads.name(), ov::Any(-1)}}));
    OV_EXPECT_THROW(
            std::ignore = ie.compile_model(model, target_device, {{ov::compilation_num_threads.name(), ov::Any(-1)}}),
            ::ov::Exception, testing::HasSubstr("ov::compilation_num_threads must be positive int32 value"));
}

INSTANTIATE_TEST_SUITE_P(nightly_BehaviorTests_OVClassGetMetricTest, OVClassGetMetricAndPrintNoThrow,
                         ::testing::Values(ov::test::utils::DEVICE_NPU), OVClassNetworkTestName::getTestCaseName);

// Several devices case
INSTANTIATE_TEST_SUITE_P(backwardDrvComp_nightly_BehaviorTests_OVClassSeveralDevicesTest, OVClassSeveralDevicesTestCompileModel,
                         ::testing::Values(std::vector<std::string>(
                                 {std::string(ov::test::utils::DEVICE_NPU) + "." +
                                  removeDeviceNameOnlyID(ov::test::utils::getTestsPlatformFromEnvironmentOr("3700"))})),
                         (ov::test::utils::appendPlatformTypeTestName<OVClassSeveralDevicesTestCompileModel>));

INSTANTIATE_TEST_SUITE_P(backwardDrvComp_nightly_BehaviorTests_OVClassModelOptionalTestP, OVClassModelOptionalTestP,
                         ::testing::Values(ov::test::utils::DEVICE_NPU),
                         (ov::test::utils::appendPlatformTypeTestName<OVClassModelOptionalTestP>));

INSTANTIATE_TEST_SUITE_P(
        backwardDrvComp_nightly_BehaviorTests_OVClassSeveralDevicesTest, OVClassSeveralDevicesTestQueryModel,
        ::testing::Values(std::vector<std::string>(
                {std::string(ov::test::utils::DEVICE_NPU) + "." +
                         removeDeviceNameOnlyID(ov::test::utils::getTestsPlatformFromEnvironmentOr("3700")),
                 std::string(ov::test::utils::DEVICE_NPU) + "." +
                         removeDeviceNameOnlyID(ov::test::utils::getTestsPlatformFromEnvironmentOr("3700"))})),
        (ov::test::utils::appendPlatformTypeTestName<OVClassSeveralDevicesTestQueryModel>));

}  // namespace
