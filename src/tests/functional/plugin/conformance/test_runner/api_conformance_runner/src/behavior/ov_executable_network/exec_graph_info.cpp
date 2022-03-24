// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "behavior/ov_executable_network/exec_graph_info.hpp"
#include "ov_api_conformance_helpers.hpp"

#include "ie_plugin_config.hpp"
#include <common_test_utils/test_constants.hpp>


using namespace ov::test::behavior;
using namespace ov::test::conformance;
namespace {
const std::vector<ov::element::Type_t> ovExecGraphInfoElemTypes = {
        ov::element::i8,
        ov::element::i16,
        ov::element::i32,
        ov::element::i64,
        ov::element::u8,
        ov::element::u16,
        ov::element::u32,
        ov::element::u64,
        ov::element::f16,
        ov::element::f32,
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVExecGraphImportExportTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(ovExecGraphInfoElemTypes),
                                 ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                 ::testing::ValuesIn(empty_ov_config)),
                         OVExecGraphImportExportTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests,
                         OVExecGraphImportExportTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(ovExecGraphInfoElemTypes),
                                 ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                 ::testing::ValuesIn(generate_ov_configs(CommonTestUtils::DEVICE_MULTI))),
                         OVExecGraphImportExportTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests,
         OVExecGraphImportExportTest,
        ::testing::Combine(
                ::testing::ValuesIn(ovExecGraphInfoElemTypes),
                ::testing::Values(CommonTestUtils::DEVICE_AUTO),
                ::testing::ValuesIn(generate_ov_configs(CommonTestUtils::DEVICE_AUTO))),
        OVExecGraphImportExportTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Hetero_BehaviorTests,
         OVExecGraphImportExportTest,
        ::testing::Combine(::testing::ValuesIn(ovExecGraphInfoElemTypes),
                           ::testing::Values(CommonTestUtils::DEVICE_HETERO),
                           ::testing::ValuesIn(generate_ov_configs(CommonTestUtils::DEVICE_HETERO))),
        OVExecGraphImportExportTest::getTestCaseName);

}  // namespace