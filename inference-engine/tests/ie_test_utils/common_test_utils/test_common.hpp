// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

namespace CommonTestUtils {

class TestsCommon : virtual public ::testing::Test {
protected:
    virtual void Run() {}

    TestsCommon();

    ~TestsCommon() override;
};

}  // namespace CommonTestUtils
