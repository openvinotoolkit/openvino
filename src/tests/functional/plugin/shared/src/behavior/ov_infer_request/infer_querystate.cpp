// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <gtest/gtest.h>
#include "behavior/ov_infer_request/infer_querystate.hpp"

namespace ov {
namespace test {
namespace behavior {
std::string OVInferRequestQueryStateTest::getTestCaseName(testing::TestParamInfo<InferRequestParams> obj) {
    std::string targetDevice;
    ov::AnyMap configuration;
    std::tie(targetDevice, configuration) = obj.param;
    std::replace(targetDevice.begin(), targetDevice.end(), ':', '.');
    std::ostringstream result;
    result << "querystate_targetDevice=" << targetDevice << "_";
    if (!configuration.empty()) {
        using namespace CommonTestUtils;
        for (auto& configItem : configuration) {
            result << "configItem=" << configItem.first << "_";
            configItem.second.print(result);
        }
    }
    return result.str();
}

TEST_P(OVInferRequestQueryStateTest, QuerystateTest) {
    req = execNet.create_infer_request();
    ASSERT_NO_THROW(req.query_state());
}

TEST_P(OVInferRequestQueryStateExceptionTest, QuerystateTestThrowNotImplemented) {
    req = execNet.create_infer_request();
    EXPECT_ANY_THROW(req.query_state());
}
}  // namespace behavior
}  // namespace test
}  // namespace ov