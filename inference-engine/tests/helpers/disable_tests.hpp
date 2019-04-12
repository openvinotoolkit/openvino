// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#undef  TEST_F
#define TEST_F(test_fixture, test_name) \
GTEST_TEST_(test_fixture, DISABLED_ ## test_name, test_fixture, \
              ::testing::internal::GetTypeId<test_fixture>())

#undef  TEST
#define TEST(a, b) GTEST_TEST(test_case_name, DISABLED_ ## test_name)