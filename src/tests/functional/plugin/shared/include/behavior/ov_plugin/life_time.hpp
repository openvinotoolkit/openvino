// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "base/ov_behavior_test_utils.hpp"

namespace ov {
namespace test {
namespace behavior {

class OVHoldersTest : public APIBaseTest,
                      public ::testing::WithParamInterface<std::string> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<std::string> obj);

    void SetUp() override;

    void TearDown() override;

protected:
    void set_api_entity() override { api_entity = utils::ov_entity::ov_plugin; }
    std::string deathTestStyle;
    std::shared_ptr<ngraph::Function> function;
};

class OVHoldersTestOnImportedNetwork : public APIBaseTest,
                                       public ::testing::WithParamInterface<std::string> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<std::string> obj);

    void SetUp() override;
    void TearDown() override;

protected:
    void set_api_entity() override { api_entity = utils::ov_entity::ov_plugin; }

    std::shared_ptr<ngraph::Function> function;
    std::string deathTestStyle;
};
}  // namespace behavior
}  // namespace test
}  // namespace ov