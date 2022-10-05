// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "base/ov_behavior_test_utils.hpp"
#include "openvino/runtime/intel_auto/properties.hpp"

namespace ov {
namespace test {
namespace behavior {

class OVHoldersTest : public OVPluginTestBase,
                      public ::testing::WithParamInterface<std::string> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<std::string> obj);
    void SetUp() override;
    void TearDown() override;

protected:
    std::string deathTestStyle;
    std::shared_ptr<ngraph::Function> function;
};

class OVHoldersTestOnImportedNetwork : public OVPluginTestBase,
                                       public ::testing::WithParamInterface<std::string> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<std::string> obj);
    void SetUp() override;
    void TearDown() override;

protected:
    std::shared_ptr<ngraph::Function> function;
    std::string deathTestStyle;
};

using OVHoldersTestWithConfig = OVHoldersTest;
}  // namespace behavior
}  // namespace test
}  // namespace ov