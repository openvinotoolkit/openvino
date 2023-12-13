// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "base/behavior_test_utils.hpp"
#include "common_test_utils/test_common.hpp"
#include "functional_test_utils/ov_plugin_cache.hpp"
#include "openvino/runtime/auto/properties.hpp"

namespace ov {
namespace test {
namespace behavior {

using memoryStateParams = std::tuple<std::shared_ptr<ngraph::Function>,  // CNNNetwork to work with
                                     std::vector<std::string>,           //  Memory States to query
                                     std::string,                        // Target device name
                                     ov::AnyMap>;                        // device configuration

class OVInferRequestVariableStateTest : public testing::WithParamInterface<memoryStateParams>,
                                        public OVInferRequestTestBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<memoryStateParams>& obj);
    void SetUp() override;
    void TearDown() override;
    static std::shared_ptr<ngraph::Function> get_network();

protected:
    std::shared_ptr<ngraph::Function> net;
    std::vector<std::string> statesToQuery;
    std::string deviceName;
    ov::AnyMap configuration;
    ov::CompiledModel prepare_network();
};
}  // namespace behavior
}  // namespace test
}  // namespace ov