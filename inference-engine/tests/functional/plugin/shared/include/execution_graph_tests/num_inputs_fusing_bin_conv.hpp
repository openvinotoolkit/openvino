// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "ngraph_functions/builders.hpp"
#include "functional_test_utils/layer_test_utils.hpp"

namespace LayerTestsDefinitions {

class ExecGraphInputsFusingBinConv : public CommonTestUtils::TestsCommon, public testing::WithParamInterface<std::string> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<std::string> obj);
    std::shared_ptr<ngraph::Function> fnPtr;
    std::string targetDevice;

protected:
    void SetUp() override;
    void TearDown() override;
};

}  // namespace LayerTestsDefinitions
