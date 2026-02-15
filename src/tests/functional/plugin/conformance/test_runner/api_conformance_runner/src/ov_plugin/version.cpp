// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_api_conformance_helpers.hpp"
#include "behavior/ov_plugin/version.hpp"

using namespace ov::test::behavior;
using namespace ov::test::conformance;

namespace {

INSTANTIATE_TEST_SUITE_P(ov_plugin_mandatory,
                         VersionTests,
                         ::testing::Values(ov::test::utils::target_device),
                         VersionTests::getTestCaseName);

}