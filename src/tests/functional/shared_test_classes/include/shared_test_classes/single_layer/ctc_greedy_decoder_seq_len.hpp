// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>
#include <string>
#include <tuple>
#include <vector>

#include "shared_test_classes/base/layer_test_utils.hpp"

namespace LayerTestsDefinitions {
typedef std::tuple<
        InferenceEngine::SizeVector,   // Input shape
        int,                           // Sequence lengths
        InferenceEngine::Precision,    // Probabilities precision
        InferenceEngine::Precision,    // Indices precision
        int,                           // Blank index
        bool,                          // Merge repeated
        std::string                    // Device name
    > ctcGreedyDecoderSeqLenParams;

class CTCGreedyDecoderSeqLenLayerTest
    :  public testing::WithParamInterface<ctcGreedyDecoderSeqLenParams>,
       virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ctcGreedyDecoderSeqLenParams>& obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
