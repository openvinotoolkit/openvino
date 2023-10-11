// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <vector>
#include <string>
#include <memory>

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ov_models/builders.hpp"

namespace ExecutionGraphTests {

std::shared_ptr<ngraph::Function> makeEltwiseFunction(const std::vector<InferenceEngine::Precision>& inputPrecisions);
std::shared_ptr<ngraph::Function> makeFakeQuantizeReluFunction(const std::vector<InferenceEngine::Precision>& inputPrecisions);
std::shared_ptr<ngraph::Function> makeFakeQuantizeBinaryConvolutionFunction(const std::vector<InferenceEngine::Precision> &inputPrecisions);

struct RuntimePrecisionSpecificParams {
    std::function<std::shared_ptr<ngraph::Function>(const std::vector<InferenceEngine::Precision>& inputPrecisions)> makeFunction;
    std::vector<InferenceEngine::Precision> inputPrecisions;
    std::map<std::string, InferenceEngine::Precision> expectedPrecisions;
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
    std::shared_ptr<ngraph::Function> fnPtr;
    std::map<std::string, InferenceEngine::Precision> expectedPrecisions;
protected:
    void SetUp() override;

    void TearDown() override;
};

}  // namespace ExecutionGraphTests
