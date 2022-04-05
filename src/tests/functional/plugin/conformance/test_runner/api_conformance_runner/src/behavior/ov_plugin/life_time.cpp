// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/life_time.hpp"
#include "ov_api_conformance_helpers.hpp"

using namespace ov::test::behavior;
using namespace ov::test::conformance;

namespace {

INSTANTIATE_TEST_SUITE_P(ov_plugin, OVHoldersTest,
        ::testing::ValuesIn(return_all_possible_device_combination()),
        OVHoldersTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(ov_plugin, OVHoldersTestOnImportedNetwork,
        ::testing::ValuesIn(return_all_possible_device_combination()),
        OVHoldersTestOnImportedNetwork::getTestCaseName);
}  // namespace
