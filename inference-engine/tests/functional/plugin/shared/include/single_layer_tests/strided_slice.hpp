// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "functional_test_utils/layer_test_utils.hpp"

namespace LayerTestsDefinitions {

struct StridedSliceSpecificParams {
    InferenceEngine::SizeVector inputShape;
    std::vector<int64_t> begin;
    std::vector<int64_t> end;
    std::vector<int64_t> strides;
    std::vector<int64_t> beginMask;
    std::vector<int64_t> endMask;
    std::vector<int64_t> newAxisMask;
    std::vector<int64_t> shrinkAxisMask;
    std::vector<int64_t> ellipsisAxisMask;
};

using StridedSliceParams = std::tuple<
        StridedSliceSpecificParams,
        InferenceEngine::Precision,        // Net precision
        InferenceEngine::Precision,        // Input precision
        InferenceEngine::Precision,        // Output precision
        InferenceEngine::Layout,           // Input layout
        InferenceEngine::Layout,           // Output layout
        std::string,                       // Device name
        std::map<std::string, std::string> // Additional network configuration
>;

class StridedSliceLayerTest : public testing::WithParamInterface<StridedSliceParams>,
                              virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<StridedSliceParams> &obj);

protected:
    void SetUp() override;
};
}  // namespace LayerTestsDefinitions
