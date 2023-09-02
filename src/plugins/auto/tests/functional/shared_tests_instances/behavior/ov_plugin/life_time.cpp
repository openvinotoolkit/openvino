// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/life_time.hpp"

using namespace ov::test::behavior;
namespace {
    INSTANTIATE_TEST_SUITE_P(smoke_VirtualPlugin_BehaviorTests, OVHoldersTest,
            ::testing::Values("AUTO:CPU",
                              "MULTI:CPU",
                              "AUTO:GPU",
                              "MULTI:GPU"),
            OVHoldersTest::getTestCaseName);

const std::vector<std::string> device_names_and_priorities = {
        "MULTI:GPU", // GPU via MULTI,
        "AUTO:GPU",  // GPU via AUTO,
        "AUTO:GPU,CPU", // GPU+CPU
        "AUTO:CPU,GPU", // CPU+GPU
        "MULTI:GPU,CPU", // GPU+CPU
        "MULTI:CPU,GPU", // CPU+GPU
};
    INSTANTIATE_TEST_SUITE_P(smoke_VirtualPlugin_BehaviorTests, OVHoldersTestWithConfig,
                    ::testing::ValuesIn(device_names_and_priorities),
            OVHoldersTestWithConfig::getTestCaseName);
}  // namespace
