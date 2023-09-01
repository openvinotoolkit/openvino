// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <memory>

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ov_models/builders.hpp"

namespace LayerTestsDefinitions {

typedef std::tuple<
        size_t,                         // Num splits
        int64_t,                        // Axis
        InferenceEngine::Precision,     // Net precision
        InferenceEngine::Precision,     // Input precision
        InferenceEngine::Precision,     // Output precision
        InferenceEngine::Layout,        // Input layout
        InferenceEngine::Layout,        // Output layout
        std::vector<size_t>,            // Input shapes
        std::vector<size_t>,            // Used outputs indices
        std::string                     // Target device name
> splitParams;

class SplitLayerTest : public testing::WithParamInterface<splitParams>,
                       virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<splitParams>& obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
