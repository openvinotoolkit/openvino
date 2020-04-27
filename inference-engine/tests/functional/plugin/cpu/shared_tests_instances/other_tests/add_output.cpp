// Copyright (C) 2020 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include "other/add_output.hpp"

const auto addOutputParams =
    ::testing::Combine(::testing::Values("Memory_1"), ::testing::Values(CommonTestUtils::DEVICE_CPU));

INSTANTIATE_TEST_CASE_P(AddOutputBasic, AddOutputTestsCommonClass, addOutputParams,
                        AddOutputTestsCommonClass::getTestCaseName);

TEST_P(AddOutputTestsCommonClass, basic) {
    run_test();
}
