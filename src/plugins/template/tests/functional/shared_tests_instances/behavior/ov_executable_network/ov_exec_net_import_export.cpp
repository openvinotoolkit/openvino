// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "behavior/compiled_model/import_export.hpp"
#include "common_test_utils/test_constants.hpp"
#include "template/properties.hpp"

using namespace ov::test::behavior;
namespace {
const std::vector<ov::element::Type_t> netPrecisions = {
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
const std::vector<ov::AnyMap> configs = {
    {ov::template_plugin::export_hsm_format(true)},
    {ov::template_plugin::export_hsm_format(false)},
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVCompiledGraphImportExportTest,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_TEMPLATE),
                                            ::testing::ValuesIn(configs)),
                         OVCompiledGraphImportExportTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVClassCompiledModelImportExportTestP,
                         ::testing::Values(ov::test::utils::DEVICE_TEMPLATE),
                         ::testing::PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVCompiledModelHsmBlobTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_TEMPLATE),
                                            ::testing::Values(ov::AnyMap{
                                                ov::template_plugin::export_hsm_format(true)})),
                         OVCompiledModelHsmBlobTest::getTestCaseName);

}  // namespace
