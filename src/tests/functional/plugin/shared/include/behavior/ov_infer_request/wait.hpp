// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "base/ov_behavior_test_utils.hpp"

namespace ov {
namespace test {
namespace behavior {
struct OVInferRequestWaitTests : public OVInferRequestTests {
    void SetUp() override;
    static std::string getTestCaseName(testing::TestParamInfo<InferRequestParams> obj);
    void TearDown() override;
    ov::InferRequest req;
    ov::Output<const ov::Node> input;
    ov::Output<const ov::Node> output;
};
}  // namespace behavior
}  // namespace test
}  // namespace ov
