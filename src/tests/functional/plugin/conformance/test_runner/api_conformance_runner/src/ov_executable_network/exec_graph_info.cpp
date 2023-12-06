// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "behavior/compiled_model/import_export.hpp"
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
        ov::element::f64,
        ov::element::bf16,
        ov::element::boolean,
};

INSTANTIATE_TEST_SUITE_P(ov_compiled_model,
                         OVCompiledGraphImportExportTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(ovExecGraphInfoElemTypes),
                                 ::testing::Values(targetDevice),
                                 ::testing::Values(pluginConfig)),
                         OVCompiledGraphImportExportTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        ov_compiled_model, OVClassCompiledModelImportExportTestP,
        ::testing::Values(targetDevice));

}  // namespace
