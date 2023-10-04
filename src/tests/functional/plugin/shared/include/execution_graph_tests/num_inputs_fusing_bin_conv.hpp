// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "ov_models/builders.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"

namespace ExecutionGraphTests {

class ExecGraphInputsFusingBinConv : public ov::test::TestsCommon, public testing::WithParamInterface<std::string> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<std::string> obj);
    std::shared_ptr<ngraph::Function> fnPtr;
    std::string targetDevice;

protected:
    void SetUp() override;
    void TearDown() override;
};

}  // namespace ExecutionGraphTests
