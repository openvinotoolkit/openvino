// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "common_test_utils/test_common.hpp"

namespace ExecutionGraphTests {

class ExecGraphInputsFusingBinConv : public ov::test::TestsCommon, public testing::WithParamInterface<std::string> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<std::string> obj);
    std::shared_ptr<ov::Model> ov_model;

protected:
    void SetUp() override;
    void TearDown() override;
};

}  // namespace ExecutionGraphTests
