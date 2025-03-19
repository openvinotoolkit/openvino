// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include <tuple>
#include <vector>
#include <string>
#include <memory>

#include "common_test_utils/test_common.hpp"

namespace ExecutionGraphTests {

std::shared_ptr<ov::Model> makeEltwiseFunction(const std::vector<ov::element::Type>& inputPrecisions);
std::shared_ptr<ov::Model> makeFakeQuantizeReluFunction(const std::vector<ov::element::Type>& inputPrecisions);
std::shared_ptr<ov::Model> makeFakeQuantizeBinaryConvolutionFunction(const std::vector<ov::element::Type> &inputPrecisions);

struct RuntimePrecisionSpecificParams {
    std::function<std::shared_ptr<ov::Model>(const std::vector<ov::element::Type>& inputPrecisions)> makeFunction;
    std::vector<ov::element::Type> inputPrecisions;
    std::map<std::string, ov::element::Type> expectedPrecisions;
};

using ExecGraphRuntimePrecisionParams = std::tuple<
    RuntimePrecisionSpecificParams,
    std::string // Target Device
>;

class ExecGraphRuntimePrecision : public testing::WithParamInterface<ExecGraphRuntimePrecisionParams>,
                                 public ov::test::TestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ExecGraphRuntimePrecisionParams> obj);
    std::string targetDevice;
    std::shared_ptr<ov::Model> fnPtr;
    std::map<std::string, ov::element::Type> expectedPrecisions;
protected:
    void SetUp() override;

    void TearDown() override;
};

}  // namespace ExecutionGraphTests
