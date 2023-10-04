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
#include "ov_models/utils/ov_helpers.hpp"

namespace SubgraphTestsDefinitions {

typedef struct {
    InferenceEngine::SizeVector kernelSize;
    InferenceEngine::SizeVector strides;
    std::vector<ptrdiff_t> padBegin;
    std::vector<ptrdiff_t> padEnd;
    size_t numOutChannels;
    InferenceEngine::SizeVector poolingWindow;
    InferenceEngine::SizeVector poolingStride;
} convReluSpecificParams;

typedef struct {
    InferenceEngine::SizeVector inputShape;
    std::vector<convReluSpecificParams> sequenceDesc;
} convReluSpecificParamsAll;

typedef std::tuple<
    convReluSpecificParamsAll,          // CNN2D sequence desc
    InferenceEngine::Precision,         // Net precision
    InferenceEngine::Precision,         // Input precision
    InferenceEngine::Precision,         // Output precision
    LayerTestsUtils::TargetDevice,      // Device name
    std::map<std::string, std::string>  // Configuration
> convReluSequenceTestParamsSet;

class ConvolutionReluSequenceTest : public testing::WithParamInterface<convReluSequenceTestParamsSet>,
                             virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<convReluSequenceTestParamsSet>& obj);

protected:
    void SetUp() override;
};

}  // namespace SubgraphTestsDefinitions
