// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"

namespace LayerTestsDefinitions {

class ExecGraphKeepAssignNode : public testing::TestWithParam<std::string> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<std::string> obj);
};

}  // namespace LayerTestsDefinitions