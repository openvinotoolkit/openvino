// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"

namespace ExecutionGraphTests {

class ExecGraphKeepAssignNode : public testing::TestWithParam<std::string> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<std::string> obj);
};

}  // namespace ExecutionGraphTests
