// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "shared_test_classes/base/layer_test_utils.hpp"

namespace LayerTestsDefinitions {

struct SliceSpecificParams {
    InferenceEngine::SizeVector inputShape;
    std::vector<int64_t> start;
    std::vector<int64_t> stop;
    std::vector<int64_t> step;
    std::vector<int64_t> axes;
};

using SliceParams = std::tuple<
        SliceSpecificParams,
        InferenceEngine::Precision,        // Net precision
        InferenceEngine::Precision,        // Input precision
        InferenceEngine::Precision,        // Output precision
        InferenceEngine::Layout,           // Input layout
        InferenceEngine::Layout,           // Output layout
        std::string,                       // Device name
        std::map<std::string, std::string> // Additional network configuration
>;

class SliceLayerTest : public testing::WithParamInterface<SliceParams>,
                              virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<SliceParams> &obj);

protected:
    void SetUp() override;
};
}  // namespace LayerTestsDefinitions
