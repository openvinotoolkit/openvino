// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "overload/compiled_model/blob_compatibility.hpp"

#include <gtest/gtest.h>

#include <map>
#include <string>

#include "common/utils.hpp"

using namespace ov::test::behavior;

const auto all_models = []() -> std::vector<std::string> {
    std::vector<std::string> models(DUMMY_MODELS.size());
    std::transform(DUMMY_MODELS.begin(),
                   DUMMY_MODELS.end(),
                   models.begin(),
                   [](const decltype(DUMMY_MODELS)::value_type& pair) {
                       return pair.second;
                   });
    return models;
}();

const auto match_platform =
    !ov::test::utils::NpuTestEnvConfig::getInstance().IE_NPU_TESTS_PLATFORM.empty()
        ? (PARSED_PLATFORMS.find(ov::test::utils::NpuTestEnvConfig::getInstance().IE_NPU_TESTS_PLATFORM) !=
                   PARSED_PLATFORMS.end()
               ? PLATFORMS.at(
                     PARSED_PLATFORMS.at(ov::test::utils::NpuTestEnvConfig::getInstance().IE_NPU_TESTS_PLATFORM))
               : "")
        : "";

const auto all_ov_releases = []() -> std::vector<std::string> {
    std::vector<std::string> ov_releases(OV_VERSIONS.size());
    std::transform(OV_VERSIONS.begin(),
                   OV_VERSIONS.end(),
                   ov_releases.begin(),
                   [](const decltype(OV_VERSIONS)::value_type& pair) {
                       return pair.second;
                   });
    return ov_releases;
}();

const auto all_drivers = []() -> std::vector<std::string> {
    std::vector<std::string> drivers(DRIVERS.size());
    std::transform(DRIVERS.begin(), DRIVERS.end(), drivers.begin(), [](const decltype(DRIVERS)::value_type& pair) {
        return pair.second;
    });
    return drivers;
}();

const auto pv_compatible_models = []() -> std::vector<std::string> {
    std::vector<std::string> models(all_models.size() - 1);
    std::copy_if(all_models.begin(), all_models.end(), models.begin(), [](const std::string& model) {
        return DUMMY_MODELS.at(E_DUMMY_MODELS::DUMMY_MODEL_DYNAMIC_SHAPES) != model;
    });
    return models;
}();

const auto all_drivers_except_pv = []() -> std::vector<std::string> {
    std::vector<std::string> drivers(all_drivers.size() - 1);
    std::copy_if(all_drivers.begin(), all_drivers.end(), drivers.begin(), [](const std::string& driver) {
        return DRIVERS.at(E_DRIVERS::DRIVER_1688) != driver;
    });
    return drivers;
}();

INSTANTIATE_TEST_SUITE_P(smoke_Behavior_NPU,
                         OVBlobCompatibilityNPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(all_models),
                                            ::testing::Values(match_platform),
                                            ::testing::ValuesIn(all_ov_releases),
                                            ::testing::ValuesIn(all_drivers_except_pv)),
                         ov::test::utils::appendPlatformTypeTestName<OVBlobCompatibilityNPU>);

#ifdef _WIN32  // Linux supports only ELF backend
INSTANTIATE_TEST_SUITE_P(smoke_Behavior_NPU,
                         OVBlobCompatibilityNPU_PV_Driver_No_Throw,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(pv_compatible_models),
                                            ::testing::Values(match_platform),
                                            ::testing::ValuesIn(all_ov_releases),
                                            ::testing::Values(DRIVERS.at(E_DRIVERS::DRIVER_1688))),
                         ov::test::utils::appendPlatformTypeTestName<OVBlobCompatibilityNPU_PV_Driver_No_Throw>);
#endif
