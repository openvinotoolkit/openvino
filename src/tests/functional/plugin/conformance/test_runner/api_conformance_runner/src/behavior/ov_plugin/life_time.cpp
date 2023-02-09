// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/life_time.hpp"
#include "ov_api_conformance_helpers.hpp"

using namespace ov::test::behavior;
using namespace ov::test::conformance;

namespace {

INSTANTIATE_TEST_SUITE_P(ov_plugin, OVLifeTimeTest,
        ::testing::ValuesIn(return_all_possible_device_combination()),
        OVLifeTimeTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(ov_plugin, OVLifeTimeTestOnImportedNetwork,
        ::testing::ValuesIn(return_all_possible_device_combination()),
        OVLifeTimeTestOnImportedNetwork::getTestCaseName);
}  // namespace
