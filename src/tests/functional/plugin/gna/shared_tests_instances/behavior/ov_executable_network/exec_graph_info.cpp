// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "behavior/ov_executable_network/ov_exec_net_import_export.hpp"

#include "ie_plugin_config.hpp"
#include <common_test_utils/test_constants.hpp>

using namespace ov::test::behavior;
namespace {
const std::vector<ov::element::Type_t> netPrecisions = {
        ov::element::i16,
        ov::element::u8,
        ov::element::f32
};
const std::vector<std::map<std::string, std::string>> configs = {
    {{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}}
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVExecNetworkImportExport,
                         ::testing::Combine(
                                 ::testing::ValuesIn(netPrecisions),
                                 ::testing::Values(CommonTestUtils::DEVICE_GNA),
                                 ::testing::ValuesIn(configs)),
                         OVExecNetworkImportExport::getTestCaseName);

}  // namespace