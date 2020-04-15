// Copyright (C) 2020 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <functional>
#include <vector>
#include <memory>

#include "ie_core.hpp"
#include "ngraph/opsets/opset1.hpp"

#include "functional_test_utils/blob_utils.hpp"
#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/layer_test_utils.hpp"

namespace NGraphConversionTestsDefinitions {

class PluginSpecificConversion : public CommonTestUtils::TestsCommon, public testing::WithParamInterface<std::string> {
public:
    std::string device;

    static std::string getTestCaseName(const testing::TestParamInfo<std::string> & obj);

protected:
    void SetUp() override;
};
}  // namespace NGraphConversionTestsDefinitions
