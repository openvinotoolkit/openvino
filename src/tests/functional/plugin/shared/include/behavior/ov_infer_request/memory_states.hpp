// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "base/ov_behavior_test_utils.hpp"
#include "common_test_utils/test_common.hpp"

namespace ov {
namespace test {
namespace behavior {

using memoryStateParams = std::tuple<std::shared_ptr<ov::Model>,  // Model to work with
                                     std::vector<std::string>,    //  Memory States to query
                                     std::string,                 // Target device name
                                     ov::AnyMap>;                 // device configuration

class OVInferRequestVariableStateTest : public testing::WithParamInterface<memoryStateParams>,
                                        public OVInferRequestTestBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<memoryStateParams>& obj);
    void SetUp() override;
    void TearDown() override;
    static std::shared_ptr<ov::Model> get_network();

protected:
    std::shared_ptr<ov::Model> net;
    std::vector<std::string> statesToQuery;
    std::string deviceName;
    ov::AnyMap configuration;
    ov::CompiledModel prepare_network();
};
}  // namespace behavior
}  // namespace test
}  // namespace ov