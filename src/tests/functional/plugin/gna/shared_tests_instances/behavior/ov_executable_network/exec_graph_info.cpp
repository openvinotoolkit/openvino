// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "behavior/ov_executable_network/exec_graph_info.hpp"

#include "ie_plugin_config.hpp"
#include <common_test_utils/test_constants.hpp>

using namespace ov::test::behavior;
namespace {
const std::vector<ov::element::Type_t> netPrecisions = {
        ov::element::i16,
        ov::element::u8,
        ov::element::f32
};
const std::vector<ov::AnyMap> configs = {
    {{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}}
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVExecGraphImportExportTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(netPrecisions),
                                 ::testing::Values(CommonTestUtils::DEVICE_GNA),
                                 ::testing::ValuesIn(configs)),
                         OVExecGraphImportExportTest::getTestCaseName);

}  // namespace