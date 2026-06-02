// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "blob_compatibility.hpp"

#include <gtest/gtest.h>

#include <map>
#include <string>

#include "common/utils.hpp"

using namespace ov::test::behavior;

const auto all_models = []() {
    std::vector<std::string> result;
    for (const auto& [key, value] : DUMMY_MODELS)
        result.push_back(value);
    return result;
}();

const auto match_platforms = []() -> std::vector<std::string> {
    const auto& platformEnv = ov::test::utils::NpuTestEnvConfig::getInstance().IE_NPU_TESTS_PLATFORM;
    if (platformEnv.empty())
        return {};
    const auto it = PARSED_PLATFORMS.find(ov::intel_npu::Platform::standardize(platformEnv));
    if (it == PARSED_PLATFORMS.end())
        return {};
    // PTL only has CIP precompiled blobs
    if (it->second == E_PLATFORMS::PTL)
        return {};
    return {PLATFORMS.at(it->second)};
}();

const auto all_ov_releases = []() {
    std::vector<std::string> result;
    for (const auto& [key, value] : OV_VERSIONS)
        result.push_back(value);
    return result;
}();

const auto pv_compatible_models = []() {
    std::vector<std::string> result;
    for (const auto& [key, value] : DUMMY_MODELS)
        if (value != DUMMY_MODELS.at(E_DUMMY_MODELS::DUMMY_MODEL_DYNAMIC_SHAPES))
            result.push_back(value);
    return result;
}();

const auto all_drivers_except_pv = []() {
    std::vector<std::string> result;
    for (const auto& [key, value] : DRIVERS)
        if (value != DRIVERS.at(E_DRIVERS::DRIVER_1688))
            result.push_back(value);
    return result;
}();

const std::vector<ov::AnyMap> config = {{}};
const std::vector<ov::AnyMap> cipConfig = {{ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::PLUGIN)}};

const std::vector<std::string> cip_ov_releases = {OV_VERSIONS.at(E_OV_VERSIONS::OV_2026_0_0)};

INSTANTIATE_TEST_SUITE_P(smoke_Behavior_NPU,
                         OVBlobCompatibilityNPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(all_models),
                                            ::testing::ValuesIn(match_platforms),
                                            ::testing::ValuesIn(all_ov_releases),
                                            ::testing::ValuesIn(all_drivers_except_pv),
                                            ::testing::ValuesIn(config)),
                         ov::test::utils::appendPlatformTypeTestName<OVBlobCompatibilityNPU>);

#ifdef _WIN32  // Linux supports only ELF backend
INSTANTIATE_TEST_SUITE_P(
    compatibility_smoke_Behavior_NPU,
    compatibility_OVBlobCompatibilityNPU_PV_Driver_No_Throw,
    ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                       ::testing::ValuesIn(pv_compatible_models),
                       ::testing::ValuesIn(match_platforms),
                       ::testing::ValuesIn(all_ov_releases),
                       ::testing::Values(DRIVERS.at(E_DRIVERS::DRIVER_1688)),
                       ::testing::ValuesIn(config)),
    ov::test::utils::appendPlatformTypeTestName<compatibility_OVBlobCompatibilityNPU_PV_Driver_No_Throw>);
#endif

INSTANTIATE_TEST_SUITE_P(compatibility_smoke_Behavior_NPU,
                         OVBlobCompatibilityNPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(all_models),
                                            ::testing::Values(PLATFORMS.at(E_PLATFORMS::PTL)),
                                            ::testing::ValuesIn(cip_ov_releases),
                                            ::testing::Values(std::string(CIP_PREFIX)),
                                            ::testing::ValuesIn(cipConfig)),
                         ov::test::utils::appendPlatformTypeTestName<OVBlobCompatibilityNPU>);
