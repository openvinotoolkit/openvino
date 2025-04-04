// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "overload/compiled_model/blob_compatibility.hpp"

#include <gtest/gtest.h>

#include <map>
#include <string>

using namespace ov::test::behavior;

enum class E_DUMMY_MODELS { DUMMY_MODEL, DUMMY_MODEL_STATEFUL, DUMMY_MODEL_DYNAMIC_SHAPES };

const std::map<E_DUMMY_MODELS, std::string> DUMMY_MODELS{
    {E_DUMMY_MODELS::DUMMY_MODEL, "dummy_model"},
    {E_DUMMY_MODELS::DUMMY_MODEL_STATEFUL, "dummy_model_stateful"},
    {E_DUMMY_MODELS::DUMMY_MODEL_DYNAMIC_SHAPES, "dummy_model_dynamic_shapes"}};

enum class E_PLATFORMS {
    MTL,
    LNL,
};

const std::map<E_PLATFORMS, std::string> PLATFORMS{{E_PLATFORMS::MTL, "MTL"}, {E_PLATFORMS::LNL, "LNL"}};

enum class E_OV_VERSIONS {
    OV_2024_6_0,
    OV_2025_0_0,
    OV_2025_1_0,
};

const std::map<E_OV_VERSIONS, std::string> OV_VERSIONS{{E_OV_VERSIONS::OV_2024_6_0, "ov_2024_6_0"},
                                                       {E_OV_VERSIONS::OV_2025_0_0, "ov_2025_0_0"},
                                                       {E_OV_VERSIONS::OV_2025_1_0, "ov_2025_1_0"}};

enum class E_DRIVERS { DRIVER_1688, DRIVER_3967 };

const std::map<E_DRIVERS, std::string> DRIVERS{{E_DRIVERS::DRIVER_1688, "driver_1688"},
                                               {E_DRIVERS::DRIVER_3967, "driver_1003967"}};

const std::vector<std::string> models = {DUMMY_MODELS.at(E_DUMMY_MODELS::DUMMY_MODEL),
                                         DUMMY_MODELS.at(E_DUMMY_MODELS::DUMMY_MODEL_STATEFUL),
                                         DUMMY_MODELS.at(E_DUMMY_MODELS::DUMMY_MODEL_DYNAMIC_SHAPES)};

const std::vector<std::string> platforms = {PLATFORMS.at(E_PLATFORMS::MTL), PLATFORMS.at(E_PLATFORMS::LNL)};

const std::vector<std::string> ov_releases = {OV_VERSIONS.at(E_OV_VERSIONS::OV_2024_6_0),
                                              OV_VERSIONS.at(E_OV_VERSIONS::OV_2025_0_0),
                                              OV_VERSIONS.at(E_OV_VERSIONS::OV_2025_1_0)};

const std::vector<std::string> drivers = {DRIVERS.at(E_DRIVERS::DRIVER_1688), DRIVERS.at(E_DRIVERS::DRIVER_3967)};

INSTANTIATE_TEST_SUITE_P(smoke_Behavior_NPU,
                         OVBlobCompatibilityNPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(models),
                                            ::testing::ValuesIn(platforms),
                                            ::testing::ValuesIn(ov_releases),
                                            ::testing::ValuesIn(drivers)),
                         ov::test::utils::appendPlatformTypeTestName<OVBlobCompatibilityNPU>);
