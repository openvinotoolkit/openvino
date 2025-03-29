// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_infer_request/infer_consistency.hpp"

#include <string>
#include <vector>

#include "common/npu_test_env_cfg.hpp"
#include "common/utils.hpp"
#include "intel_npu/npu_private_properties.hpp"

using namespace ov::test::behavior;

namespace {
// for deviceConfigs, the deviceConfigs[0] is target device which need to be tested.
// deviceConfigs[1], deviceConfigs[2],deviceConfigs[n] are the devices which will
// be compared with target device, the result of target should be in one of the compared
// device.
using Configs = std::vector<std::pair<std::string, ov::AnyMap>>;

auto configs = []() {
    return std::vector<Configs>{{{ov::test::utils::DEVICE_NPU, {}}}};
}();

auto autoConfigs = []() {
    return std::vector<Configs>{{{ov::test::utils::DEVICE_AUTO + std::string(":") + ov::test::utils::DEVICE_NPU,
                                  {ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)}},
                                 {ov::test::utils::DEVICE_NPU, {}}},
                                {{ov::test::utils::DEVICE_AUTO + std::string(":") + ov::test::utils::DEVICE_NPU,
                                  {ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)}},
                                 {ov::test::utils::DEVICE_NPU, {}}},
                                {{ov::test::utils::DEVICE_AUTO + std::string(":") + ov::test::utils::DEVICE_NPU,
                                  {ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT)}},
                                 {ov::test::utils::DEVICE_NPU, {}}},
                                {{ov::test::utils::DEVICE_AUTO + std::string(":") + ov::test::utils::DEVICE_NPU + "," +
                                      ov::test::utils::DEVICE_CPU,
                                  {ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)}},
                                 {ov::test::utils::DEVICE_NPU, {}},
                                 {ov::test::utils::DEVICE_CPU, {}}},
                                {{ov::test::utils::DEVICE_AUTO + std::string(":") + ov::test::utils::DEVICE_NPU + "," +
                                      ov::test::utils::DEVICE_CPU,
                                  {ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)}},
                                 {ov::test::utils::DEVICE_NPU, {}},
                                 {ov::test::utils::DEVICE_CPU, {}}},
                                {{ov::test::utils::DEVICE_AUTO + std::string(":") + ov::test::utils::DEVICE_NPU + "," +
                                      ov::test::utils::DEVICE_CPU,
                                  {ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT)}},
                                 {ov::test::utils::DEVICE_NPU, {}},
                                 {ov::test::utils::DEVICE_CPU, {}}},
                                {{ov::test::utils::DEVICE_AUTO + std::string(":") + ov::test::utils::DEVICE_CPU + "," +
                                      ov::test::utils::DEVICE_NPU,
                                  {ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT)}},
                                 {ov::test::utils::DEVICE_CPU, {}},
                                 {ov::test::utils::DEVICE_NPU, {}}}};
}();

auto multiConfigs = []() {
    return std::vector<Configs>{{{ov::test::utils::DEVICE_MULTI + std::string(":") + ov::test::utils::DEVICE_NPU,
                                  {ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)}},
                                 {ov::test::utils::DEVICE_NPU, {}}},
                                {{ov::test::utils::DEVICE_MULTI + std::string(":") + ov::test::utils::DEVICE_NPU,
                                  {ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)}},
                                 {ov::test::utils::DEVICE_NPU, {}}},
                                {{ov::test::utils::DEVICE_MULTI + std::string(":") + ov::test::utils::DEVICE_NPU + "," +
                                      ov::test::utils::DEVICE_CPU,
                                  {ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)}},
                                 {ov::test::utils::DEVICE_NPU, {}},
                                 {ov::test::utils::DEVICE_CPU, {}}},
                                {{ov::test::utils::DEVICE_MULTI + std::string(":") + ov::test::utils::DEVICE_NPU + "," +
                                      ov::test::utils::DEVICE_CPU,
                                  {ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)}},
                                 {ov::test::utils::DEVICE_NPU, {}},
                                 {ov::test::utils::DEVICE_CPU, {}}}};
}();

// 3x5 configuration takes ~65 seconds to run, which is already pretty long time
INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_BehaviorTests,
                         OVInferConsistencyTest,
                         ::testing::Combine(::testing::Values(3),  // inferRequest num
                                            ::testing::Values(5),  // infer counts
                                            ::testing::ValuesIn(configs)),
                         ov::test::utils::appendPlatformTypeTestName<OVInferConsistencyTest>);

INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_Auto_BehaviorTests,
                         OVInferConsistencyTest,
                         ::testing::Combine(::testing::Values(3),  // inferRequest num
                                            ::testing::Values(5),  // infer counts
                                            ::testing::ValuesIn(autoConfigs)),
                         ov::test::utils::appendPlatformTypeTestName<OVInferConsistencyTest>);

INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_Multi_BehaviorTests,
                         OVInferConsistencyTest,
                         ::testing::Combine(::testing::Values(3),  // inferRequest num
                                            ::testing::Values(5),  // infer counts
                                            ::testing::ValuesIn(multiConfigs)),
                         ov::test::utils::appendPlatformTypeTestName<OVInferConsistencyTest>);
}  // namespace
