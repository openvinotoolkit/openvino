// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "gtest/gtest.h"
#include "functional_test_utils/skip_tests_config.hpp"

// ======================= ExtractorsManagerTest Unit tests =======================
class SubgraphsDumperBaseTest : public ::testing::Test {
protected:
    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
    }
};
