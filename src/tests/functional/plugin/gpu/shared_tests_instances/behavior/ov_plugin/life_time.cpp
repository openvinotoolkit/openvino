// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/life_time.hpp"

using namespace ov::test::behavior;
namespace {
    INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVHoldersTest,
            ::testing::Values(CommonTestUtils::DEVICE_GPU),
            OVHoldersTest::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_VirtualPlugin_BehaviorTests, OVHoldersTest,
                    ::testing::Values("AUTO:GPU",
                                        "MULTI:GPU",
                                        //CommonTestUtils::DEVICE_BATCH,
                                        "HETERO:GPU"),
            OVHoldersTest::getTestCaseName);

const std::vector<std::string> device_names_and_priorities = {
        "MULTI:GPU", // GPU via MULTI,
        "AUTO:GPU",  // GPU via AUTO,
#ifdef ENABLE_INTEL_CPU
        "AUTO:GPU,CPU", // GPU+CPU
        "AUTO:CPU,GPU", // CPU+GPU
        "MULTI:GPU,CPU", // GPU+CPU
        "MULTI:CPU,GPU", // CPU+GPU
#endif
};
    INSTANTIATE_TEST_SUITE_P(smoke_VirtualPlugin_BehaviorTests, OVHoldersTestWithConfig,
                    ::testing::ValuesIn(device_names_and_priorities),
            OVHoldersTestWithConfig::getTestCaseName);
}  // namespace