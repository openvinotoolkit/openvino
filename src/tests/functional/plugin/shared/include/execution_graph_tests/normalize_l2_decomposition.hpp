// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "gtest/gtest.h"

namespace ExecutionGraphTests {

class ExecGrapDecomposeNormalizeL2 : public testing::TestWithParam<std::string> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<std::string> obj);
};

}  // namespace ExecutionGraphTests
