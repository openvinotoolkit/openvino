//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "behavior/ov_infer_request/perf_counters.hpp"
#include "common/utils.hpp"
#include "common/vpu_test_env_cfg.hpp"
#include "intel_npu/al/config/common.hpp"

using namespace ov::test::behavior;

namespace {
const std::vector<ov::AnyMap> configs = {
        {ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::MLIR),
         ov::intel_npu::platform(ov::test::utils::getTestsPlatformCompilerInPlugin())}};

const std::vector<ov::AnyMap> multiConfigs = {
        {ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::MLIR),
         ov::intel_npu::platform(ov::test::utils::getTestsPlatformCompilerInPlugin()),
         ov::device::priorities(ov::test::utils::DEVICE_NPU),
         ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)},
        {ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::MLIR),
         ov::intel_npu::platform(ov::test::utils::getTestsPlatformCompilerInPlugin()),
         ov::device::priorities(ov::test::utils::DEVICE_NPU),
         ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)}};

const std::vector<ov::AnyMap> autoConfigs = {
        {ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::MLIR),
         ov::intel_npu::platform(ov::test::utils::getTestsPlatformCompilerInPlugin()),
         ov::device::priorities(ov::test::utils::DEVICE_NPU),
         ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)},
        {ov::device::priorities(ov::test::utils::DEVICE_NPU),
         ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
         ov::intel_npu::platform(ov::test::utils::getTestsPlatformCompilerInPlugin())}};

// Driver compiler type config
const std::vector<ov::AnyMap> driverCompilerConfigs = {
        {ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::DRIVER)}};

const std::vector<ov::AnyMap> driverCompilerMultiConfigs = {
        {ov::device::priorities(ov::test::utils::DEVICE_NPU),
         ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT),
         ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::DRIVER)},
        {ov::device::priorities(ov::test::utils::DEVICE_NPU),
         ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
         ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::DRIVER)}};

const std::vector<ov::AnyMap> driverCompilerAutoConfigs = {
        {ov::device::priorities(ov::test::utils::DEVICE_NPU),
         ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT),
         ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::DRIVER)},
        {ov::device::priorities(ov::test::utils::DEVICE_NPU),
         ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
         ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::DRIVER)}};

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, OVInferRequestPerfCountersTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_MULTI),
                                            ::testing::ValuesIn(multiConfigs)),
                         ov::test::utils::appendPlatformTypeTestName<OVInferRequestPerfCountersTest>);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, OVInferRequestPerfCountersTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_AUTO),
                                            ::testing::ValuesIn(autoConfigs)),
                         ov::test::utils::appendPlatformTypeTestName<OVInferRequestPerfCountersTest>);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVInferRequestPerfCountersTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(configs)),
                         ov::test::utils::appendPlatformTypeTestName<OVInferRequestPerfCountersTest>);

// Driver compiler type test suite
INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests_Driver, OVInferRequestPerfCountersTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_MULTI),
                                            ::testing::ValuesIn(driverCompilerMultiConfigs)),
                         ov::test::utils::appendPlatformTypeTestName<OVInferRequestPerfCountersTest>);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests_Driver, OVInferRequestPerfCountersTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_AUTO),
                                            ::testing::ValuesIn(driverCompilerAutoConfigs)),
                         ov::test::utils::appendPlatformTypeTestName<OVInferRequestPerfCountersTest>);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests_Driver, OVInferRequestPerfCountersTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(driverCompilerConfigs)),
                         ov::test::utils::appendPlatformTypeTestName<OVInferRequestPerfCountersTest>);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVInferRequestPerfCountersExceptionTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(configs)),
                         ov::test::utils::appendPlatformTypeTestName<OVInferRequestPerfCountersExceptionTest>);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, OVInferRequestPerfCountersExceptionTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_AUTO),
                                            ::testing::ValuesIn(autoConfigs)),
                         ov::test::utils::appendPlatformTypeTestName<OVInferRequestPerfCountersExceptionTest>);

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, OVInferRequestPerfCountersExceptionTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_MULTI),
                                            ::testing::ValuesIn(multiConfigs)),
                         ov::test::utils::appendPlatformTypeTestName<OVInferRequestPerfCountersExceptionTest>);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests_Driver, OVInferRequestPerfCountersExceptionTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(driverCompilerConfigs)),
                         ov::test::utils::appendPlatformTypeTestName<OVInferRequestPerfCountersExceptionTest>);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests_Driver, OVInferRequestPerfCountersExceptionTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_AUTO),
                                            ::testing::ValuesIn(driverCompilerAutoConfigs)),
                         ov::test::utils::appendPlatformTypeTestName<OVInferRequestPerfCountersExceptionTest>);

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests_Driver, OVInferRequestPerfCountersExceptionTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_MULTI),
                                            ::testing::ValuesIn(driverCompilerMultiConfigs)),
                         ov::test::utils::appendPlatformTypeTestName<OVInferRequestPerfCountersExceptionTest>);

}  // namespace
