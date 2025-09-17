// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "internal/overload/compiled_model/property.hpp"

#include <openvino/runtime/intel_npu/properties.hpp>
#include <vector>

#include "behavior/compiled_model/properties.hpp"
#include "common/npu_test_env_cfg.hpp"
#include "common/utils.hpp"
#include "common_test_utils/subgraph_builders/conv_pool_relu.hpp"
#include "intel_npu/config/options.hpp"

using namespace ov::test::behavior;

namespace {

template <typename T>
constexpr std::vector<T> operator+(const std::vector<T>& vec1, const std::vector<T>& vec2) {
    std::vector<T> result;
    result.insert(result.end(), vec1.begin(), vec1.end());
    result.insert(result.end(), vec2.begin(), vec2.end());
    return result;
}

std::vector<std::pair<std::string, ov::Any>> exe_network_supported_properties = {
    {ov::hint::num_requests.name(), ov::Any(8)},
    {ov::hint::enable_cpu_pinning.name(), ov::Any(true)},
    {ov::hint::performance_mode.name(), ov::Any(ov::hint::PerformanceMode::THROUGHPUT)},
    {ov::hint::model_priority.name(), ov::Any(ov::hint::Priority::MEDIUM)},
    {ov::optimal_number_of_infer_requests.name(), ov::Any(2)},
};

std::vector<std::pair<std::string, ov::Any>> exe_network_immutable_properties = {
    {std::make_pair(ov::optimal_number_of_infer_requests.name(), ov::Any(2))},
    {std::make_pair(ov::hint::enable_cpu_pinning.name(), ov::Any(false))},
    {std::make_pair(ov::supported_properties.name(), ov::Any("deadbeef"))},
    {std::make_pair(ov::model_name.name(), ov::Any("deadbeef"))},
    {ov::hint::model.name(), ov::Any(std::shared_ptr<const ov::Model>(nullptr))},
    {ov::hint::model.name(),
     ov::Any(std::shared_ptr<ov::Model>(nullptr))},  // intentionally copied above to test constness
};

std::vector<std::pair<std::string, ov::Any>> plugin_public_mutable_properties = {
    {ov::hint::num_requests.name(), ov::Any(5)},
    {ov::enable_profiling.name(), ov::Any(true)},
    {ov::compilation_num_threads.name(), ov::Any(1)},
    {ov::hint::performance_mode.name(), ov::Any(ov::hint::PerformanceMode::THROUGHPUT)},
    {ov::hint::enable_cpu_pinning.name(), ov::Any(true)},
    {ov::log::level.name(), ov::Any(ov::log::Level::ERR)},
    {ov::device::id.name(), ov::Any(ov::test::utils::getDeviceNameID(ov::test::utils::getDeviceName()))},
};

std::vector<std::pair<std::string, ov::Any>> compat_plugin_internal_mutable_properties = {
    {ov::intel_npu::compilation_mode_params.name(), ov::Any("use-user-precision=false propagate-quant-dequant=0")},
    {ov::intel_npu::dma_engines.name(), ov::Any(1)},
    {ov::intel_npu::platform.name(), ov::Any(ov::intel_npu::Platform::AUTO_DETECT)},
    {ov::intel_npu::compilation_mode.name(), ov::Any("DefaultHW")},
    {ov::intel_npu::defer_weights_load.name(), ov::Any(true)},
};

std::vector<std::pair<std::string, ov::Any>> plugin_internal_mutable_properties = {
    {ov::intel_npu::max_tiles.name(), ov::Any(8)},
    {ov::intel_npu::stepping.name(), ov::Any(4)},
};

std::vector<std::pair<std::string, ov::Any>> plugin_public_immutable_properties = {
    {ov::device::uuid.name(), ov::Any("deadbeef")},
    {ov::supported_properties.name(), {ov::device::full_name.name()}},
    {ov::num_streams.name(), ov::Any(ov::streams::Num(4))},
    {ov::available_devices.name(), ov::Any(std::vector<std::string>{"deadbeef"})},
    {ov::device::capabilities.name(), ov::Any(std::vector<std::string>{"deadbeef"})},
    {ov::range_for_async_infer_requests.name(),
     ov::Any(std::tuple<unsigned int, unsigned int, unsigned int>{0, 10, 1})},
    {ov::range_for_streams.name(), ov::Any(std::tuple<unsigned int, unsigned int>{0, 10})},
    {ov::optimal_number_of_infer_requests.name(), ov::Any(4)},
    {ov::intel_npu::device_alloc_mem_size.name(), ov::Any(2)},
    {ov::intel_npu::device_total_mem_size.name(), ov::Any(2)},
};

std::vector<std::pair<std::string, ov::Any>> invalid_device_ids = {
    {ov::device::id.name(), "NPU.1"},
    {ov::device::id.name(), "NPU.-1"},
    {ov::device::id.name(), "NPU.3990"},
    {ov::device::id.name(), "NPU.DUMMY"},
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests_ClassExecutableNetworkGetPropertiesTestNPU,
                         ClassExecutableNetworkTestSuite1NPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::getDeviceName()),
                                            ::testing::ValuesIn(exe_network_supported_properties)),
                         ClassExecutableNetworkTestSuite1NPU::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests_ClassExecutableNetworkTestSuite2NPU,
                         ClassExecutableNetworkTestSuite2NPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::getDeviceName()),
                                            ::testing::ValuesIn(exe_network_immutable_properties)),
                         ClassExecutableNetworkTestSuite2NPU::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(compatibility_smoke_BehaviorTests_ClassPluginPropertiesTestNPU,
                         ClassPluginPropertiesTestSuite0NPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::getDeviceName()),
                                            ::testing::ValuesIn(plugin_public_mutable_properties)),
                         ClassPluginPropertiesTestNPU::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(compatibility_smoke_BehaviorTests_ClassPluginPropertiesTestNPU,
                         ClassPluginPropertiesTestSuite1NPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::getDeviceName()),
                                            ::testing::ValuesIn(compat_plugin_internal_mutable_properties)),
                         ClassPluginPropertiesTestNPU::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests_ClassPluginPropertiesTestNPU,
                         ClassPluginPropertiesTestSuite1NPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::getDeviceName()),
                                            ::testing::ValuesIn(plugin_internal_mutable_properties)),
                         ClassPluginPropertiesTestNPU::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests_ClassPluginPropertiesTest,
                         ClassPluginPropertiesTestSuite2NPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::getDeviceName()),
                                            ::testing::ValuesIn(plugin_public_immutable_properties)),
                         ClassPluginPropertiesTestNPU::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests_ClassPluginPropertiesOptsTest1NPU,
                         ClassPluginPropertiesTestSuite3NPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::getDeviceName()),
                                            ::testing::ValuesIn(plugin_public_immutable_properties)),
                         ClassPluginPropertiesTestNPU::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(compatibility_smoke_BehaviorTests_ClassPluginPropertiesOptsTest2NPU,
                         ClassPluginPropertiesTestSuite3NPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::getDeviceName()),
                                            ::testing::ValuesIn(plugin_public_mutable_properties)),
                         ClassPluginPropertiesTestNPU::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests_ClassPluginPropertiesOptsTest5NPU,
                         ClassPluginPropertiesTestSuite5NPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::getDeviceName()),
                                            ::testing::ValuesIn(plugin_public_mutable_properties +
                                                                compat_plugin_internal_mutable_properties +
                                                                plugin_internal_mutable_properties)),
                         ClassPluginPropertiesTestNPU::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_BehaviorTests_ClassExecutableNetworkGetPropertiesTestNPU,
    ClassPluginPropertiesTestSuite4NPU,
    ::testing::Combine(::testing::Values(ov::test::utils::getDeviceName()),
                       ::testing::ValuesIn({std::make_pair<std::string, ov::Any>("THISCONFIGKEYNOTEXIST",
                                                                                 ov::Any("THISCONFIGVALUENOTEXIST"))})),
    ClassPluginPropertiesTestSuite4NPU::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests_ClassExecutableNetworkInvalidDeviceIDTestNPU,
                         ClassExecutableNetworkInvalidDeviceIDTestSuite,
                         ::testing::Combine(::testing::Values(ov::test::utils::getDeviceName()),
                                            ::testing::ValuesIn(invalid_device_ids)),
                         ClassExecutableNetworkInvalidDeviceIDTestSuite::getTestCaseName);

}  // namespace
