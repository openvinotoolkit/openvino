// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "postpone_compiler_load.hpp"

#include "common/utils.hpp"

namespace {

template <typename T>
constexpr std::vector<T> operator+(const std::vector<T>& vec1, const std::vector<T>& vec2) {
    std::vector<T> result;
    result.insert(result.end(), vec1.begin(), vec1.end());
    result.insert(result.end(), vec2.begin(), vec2.end());
    return result;
}

std::vector<std::string> convertToStringVec(const std::vector<ov::AnyMap>& anyMapVec) {
    std::vector<std::string> retVec;
    for (const auto& anyMap : anyMapVec) {
        retVec.push_back(anyMap.begin()->first);
    }
    return retVec;
}

std::vector<ov::AnyMap> CartesianProductAnyMapVec(const std::vector<ov::AnyMap>& anyMapVec1,
                                                  const std::vector<ov::AnyMap>& anyMapVec2,
                                                  const std::vector<ov::AnyMap>& anyMapVec3) {
    std::vector<ov::AnyMap> retAnyMapVec = {};
    for (const auto& anyMap1 : anyMapVec1) {
        for (const auto& anyMap2 : anyMapVec2) {
            for (const auto& anyMap3 : anyMapVec3) {
                retAnyMapVec.push_back({*anyMap1.begin(), *anyMap2.begin(), *anyMap3.begin()});
            }
        }
    }
    return retAnyMapVec;
}

const std::vector<ov::AnyMap> runTimeProperties = {
    {{ov::intel_npu::defer_weights_load.name(), true}},
    {{ov::intel_npu::compiler_type.name(), ov::intel_npu::CompilerType::MLIR}},
};

const std::vector<ov::AnyMap> compileTimeProperties = {
    {{ov::intel_npu::qdq_optimization.name(), true}},
};

const std::vector<ov::AnyMap> bothProperties = {
    {{ov::hint::performance_mode.name(), ov::hint::PerformanceMode::THROUGHPUT}},
};

const std::vector<ov::AnyMap> metrics = {
    {{ov::device::architecture.name(), "3720"}},
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVPostponeCompilerLoadTestsNPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(CartesianProductAnyMapVec(runTimeProperties,
                                                                                          compileTimeProperties,
                                                                                          bothProperties))),
                         ov::test::utils::appendPlatformTypeTestName<OVPostponeCompilerLoadTestsNPU>);

INSTANTIATE_TEST_SUITE_P(
    smoke_BehaviorTests,
    OVRunTimePropertiesArgumentsTestsNPU,
    ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                       ::testing::ValuesIn(convertToStringVec(runTimeProperties) + convertToStringVec(metrics)),
                       ::testing::ValuesIn(runTimeProperties + compileTimeProperties + bothProperties)),
    ov::test::utils::appendPlatformTypeTestName<OVRunTimePropertiesArgumentsTestsNPU>);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVSetRunTimePropertiesTestsNPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(runTimeProperties)),
                         ov::test::utils::appendPlatformTypeTestName<OVSetRunTimePropertiesTestsNPU>);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVCompileTimePropertiesArgumentsTestsNPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(convertToStringVec(compileTimeProperties)),
                                            ::testing::ValuesIn(runTimeProperties + compileTimeProperties +
                                                                bothProperties)),
                         ov::test::utils::appendPlatformTypeTestName<OVCompileTimePropertiesArgumentsTestsNPU>);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVSetCompileTimePropertiesTestsNPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(compileTimeProperties)),
                         ov::test::utils::appendPlatformTypeTestName<OVSetCompileTimePropertiesTestsNPU>);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVBothPropertiesArgumentsTestsNPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(convertToStringVec(bothProperties)),
                                            ::testing::ValuesIn(runTimeProperties + compileTimeProperties +
                                                                bothProperties)),
                         ov::test::utils::appendPlatformTypeTestName<OVBothPropertiesArgumentsTestsNPU>);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVSetBothPropertiesTestsNPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(bothProperties)),
                         ov::test::utils::appendPlatformTypeTestName<OVSetBothPropertiesTestsNPU>);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVSetCartesianProductPropertiesTestsNPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(CartesianProductAnyMapVec(runTimeProperties,
                                                                                          compileTimeProperties,
                                                                                          bothProperties))),
                         ov::test::utils::appendPlatformTypeTestName<OVSetCartesianProductPropertiesTestsNPU>);

}  // namespace
