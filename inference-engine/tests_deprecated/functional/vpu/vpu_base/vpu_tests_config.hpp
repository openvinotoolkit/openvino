// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <string>

#include <gtest/gtest.h>

#include "functional_test_utils/skip_tests_config.hpp"

namespace vpu {
namespace tests {

const char* pluginName();
const char* pluginNameShort();
const char* deviceName();
bool deviceForceReset();

}  // namespace tests
}  // namespace vpu

// IE macro forcing gave us no ability to pass device name as variable.
// So we create this two replacements to PLUGING_CASE_WITH_SUFFIX.
#define VPU_PLUGING_CASE_WITH_SUFFIX(_suffix, _test, _params) \
    INSTANTIATE_TEST_SUITE_P(VPU_run##_suffix, _test, ::testing::Combine(::testing::Values(::vpu::tests::deviceName()), _params) )

#define DISABLED_VPU_PLUGING_CASE_WITH_SUFFIX(_suffix, _test, _params) \
    INSTANTIATE_TEST_SUITE_P(DISABLED_VPU_run##_suffix, _test, ::testing::Combine(::testing::Values(::vpu::tests::deviceName()), _params) )
