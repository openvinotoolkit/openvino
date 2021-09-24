// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <chrono>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "base/ov_behavior_test_utils.hpp"
#include "common_test_utils/test_constants.hpp"
#include "openvino/runtime/runtime.hpp"

namespace ov {
namespace test {
using OVExecNetwork = ov::test::BehaviorTestsBasic;

// Load correct network to Plugin to get executable network
TEST_P(OVExecNetwork, getInputFromFunctionWithSingleInput) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::runtime::ExecutableNetwork execNet;
    ASSERT_NO_THROW(execNet = ie->compile_model(function, targetDevice, configuration));
    ASSERT_EQ(function->inputs().size(), 1);
    ASSERT_EQ(function->inputs().size(), execNet.inputs().size());
    ASSERT_NO_THROW(execNet.input());
    ASSERT_EQ(function->input().get_tensor().get_names(), execNet.input().get_tensor().get_names());
}

}  // namespace test
}  // namespace ov
