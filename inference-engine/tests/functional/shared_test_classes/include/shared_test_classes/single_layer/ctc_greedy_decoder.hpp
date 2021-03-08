// Copyright (C) 2020-2021 Intel Corporation
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
    InferenceEngine::Precision,
    InferenceEngine::Precision,    // Input precision
    InferenceEngine::Precision,    // Output precision
    InferenceEngine::Layout,       // Input layout
    InferenceEngine::Layout,       // Output layout
    InferenceEngine::SizeVector,
    bool,
    std::string> ctcGreedyDecoderParams;

class CTCGreedyDecoderLayerTest
    :  public testing::WithParamInterface<ctcGreedyDecoderParams>,
       virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ctcGreedyDecoderParams>& obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
