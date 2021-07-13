// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <pugixml.hpp>
#include "gtest/gtest.h"

namespace ExecutionGraphTests {

class ExecGraphDumpConstantLayers : public testing::TestWithParam<std::string> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<std::string> obj);
};

} // namespace ExecutionGraphTests
