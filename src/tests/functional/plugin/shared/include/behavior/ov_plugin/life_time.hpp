// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "base/ov_behavior_test_utils.hpp"
#include "openvino/runtime/auto/properties.hpp"

namespace ov {
namespace test {
namespace behavior {

class OVLifeTimeTest : public OVPluginTestBase,
                      public ::testing::WithParamInterface<std::string> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<std::string> obj);
    void SetUp() override;
    void TearDown() override;

protected:
    std::string deathTestStyle;
    std::shared_ptr<ngraph::Function> function;
};

class OVLifeTimeTestOnImportedNetwork : public OVPluginTestBase,
                                       public ::testing::WithParamInterface<std::string> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<std::string> obj);
    void SetUp() override;
    void TearDown() override;

protected:
    std::shared_ptr<ngraph::Function> function;
    std::string deathTestStyle;
};

using OVLifeTimeTestWithConfig = OVLifeTimeTest;
}  // namespace behavior
}  // namespace test
}  // namespace ov
