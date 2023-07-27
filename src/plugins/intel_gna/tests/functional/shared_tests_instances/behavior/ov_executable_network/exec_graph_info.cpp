// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <common_test_utils/test_constants.hpp>

#include "behavior/compiled_model/import_export.hpp"
#include "ie_plugin_config.hpp"

using namespace ov::test::behavior;
namespace {
const std::vector<ov::element::Type_t> netPrecisions = {ov::element::i16, ov::element::u8, ov::element::f32};
const std::vector<ov::AnyMap> configs = {{{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}}};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVCompiledGraphImportExportTest,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs)),
                         OVCompiledGraphImportExportTest::getTestCaseName);

}  // namespace
