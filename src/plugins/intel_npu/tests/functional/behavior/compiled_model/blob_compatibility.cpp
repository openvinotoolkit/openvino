// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "overload/compiled_model/blob_compatibility.hpp"

#include <gtest/gtest.h>

using namespace ov::test::behavior;

const std::vector<std::string> models = {"dummy_model",
                                         "dummy_model_stateful",
                                         "dummy_model_dynamic_shapes"};  // fixed prefix
const std::vector<std::string> platforms = {"MTL", "LNL"};
const std::vector<std::string> ov_releases = {"ov_2024_6_0", "ov_2025_0_0", "ov_2025_1_0"};
const std::vector<std::string> drivers = {"driver_1688", "driver_1003967"};

INSTANTIATE_TEST_SUITE_P(smoke_Behavior_NPU,
                         OVBlobCompatibilityNPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(models),
                                            ::testing::ValuesIn(platforms),
                                            ::testing::ValuesIn(ov_releases),
                                            ::testing::ValuesIn(drivers)),
                         ov::test::utils::appendPlatformTypeTestName<OVBlobCompatibilityNPU>);
