// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "overload/compiled_model/import_export.hpp"

#include <gtest/gtest.h>

#include <map>
#include <string>

#include "common/utils.hpp"

using namespace ov::test::behavior;

const std::vector<ov::AnyMap> compiledModelConfigs = {{}};

INSTANTIATE_TEST_SUITE_P(
    smoke_Behavior_NPU,
    OVCompiledGraphImportExportTestNPU,
    ::testing::Combine(::testing::Values(ov::element::f16 /* not used in internal import_export tests so far */),
                       ::testing::Values(ov::test::utils::DEVICE_NPU),
                       ::testing::ValuesIn(compiledModelConfigs)),
    ov::test::utils::appendPlatformTypeTestName<OVCompiledGraphImportExportTestNPU>);
