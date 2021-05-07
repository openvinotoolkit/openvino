// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include <string>

namespace CommonTestUtils {

class TestsCommon : virtual public ::testing::Test {
protected:
    TestsCommon();
    ~TestsCommon() override;

    static std::string GetTimestamp();
    std::string GetTestName() const;
};

}  // namespace CommonTestUtils
