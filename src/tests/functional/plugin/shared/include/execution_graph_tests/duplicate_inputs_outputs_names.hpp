// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"

namespace ExecutionGraphTests {

class ExecGraphDuplicateInputsOutputsNames
    : public testing::TestWithParam<std::string> {
public:
  static std::string getTestCaseName(testing::TestParamInfo<std::string> obj);
};

} // namespace ExecutionGraphTests
