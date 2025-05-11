// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "base/ov_behavior_test_utils.hpp"
#include "openvino/runtime/auto/properties.hpp"

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
    std::shared_ptr<ov::Model> function;
};

class OVHoldersTestOnImportedNetwork : public OVPluginTestBase,
                                       public ::testing::WithParamInterface<std::string> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<std::string> obj);
    void SetUp() override;
    void TearDown() override;

protected:
    std::shared_ptr<ov::Model> function;
    std::string deathTestStyle;
};

using OVHoldersTestWithConfig = OVHoldersTest;
}  // namespace behavior
}  // namespace test
}  // namespace ov
