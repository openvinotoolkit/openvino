// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifcorer: Apache-2.0
//

#include <string>
#include <utility>
#include <vector>

#include "behavior/ov_plugin/core_integration_sw.hpp"
#include "behavior/ov_plugin/query_model.hpp"

using namespace ov::test::behavior;

namespace {

//
// OV Class Common tests with <pluginName, device_name params>
//

INSTANTIATE_TEST_SUITE_P(smoke_OVClassModelTestP,
                         OVClassModelTestP,
                         ::testing::Values(ov::test::utils::DEVICE_TEMPLATE));

INSTANTIATE_TEST_SUITE_P(smoke_OVClassModelOptionalTestP,
                         OVClassModelOptionalTestP,
                         ::testing::Values(ov::test::utils::DEVICE_TEMPLATE));

TEST(OVClassBasicPropsTest, smoke_TEMPLATEGetSetConfigNoThrow) {
    ov::Core core = ov::test::utils::create_core();

    auto device_name = ov::test::utils::DEVICE_TEMPLATE;

    for (auto&& property : core.get_property(device_name, ov::supported_properties)) {
        if (ov::device::id == property) {
            std::cout << ov::device::id.name() << " : " << core.get_property(device_name, ov::device::id) << std::endl;
        } else if (ov::enable_profiling == property) {
            std::cout << ov::enable_profiling.name() << " : " << core.get_property(device_name, ov::enable_profiling)
                      << std::endl;
        } else if (ov::hint::performance_mode == property) {
            std::cout << "Default " << ov::hint::performance_mode.name() << " : "
                      << core.get_property(device_name, ov::hint::performance_mode) << std::endl;
            core.set_property(device_name, ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
            ASSERT_EQ(ov::hint::PerformanceMode::LATENCY, core.get_property(device_name, ov::hint::performance_mode));
            core.set_property(device_name, ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT));
            ASSERT_EQ(ov::hint::PerformanceMode::THROUGHPUT,
                      core.get_property(device_name, ov::hint::performance_mode));
        }
    }
}

// OV Class Query network

INSTANTIATE_TEST_SUITE_P(smoke_OVClassQueryModelTest,
                         OVClassQueryModelTest,
                         ::testing::Values(ov::test::utils::DEVICE_TEMPLATE));

}  // namespace
