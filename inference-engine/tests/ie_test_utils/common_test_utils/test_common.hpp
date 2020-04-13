// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

namespace CommonTestUtils {

class TestsCommon : public ::testing::Test {
protected:
    TestsCommon();

    ~TestsCommon() override;
};

}  // namespace CommonTestUtils
