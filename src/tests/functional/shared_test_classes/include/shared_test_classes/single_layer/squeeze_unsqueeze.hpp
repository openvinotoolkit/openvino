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
using ShapeAxesTuple = std::pair<std::vector<size_t>, std::vector<int>>;

typedef std::tuple<
        ShapeAxesTuple,                 // InputShape (required), Squeeze indexes (if empty treated as non-existent)
        ngraph::helpers::SqueezeOpType, // OpType
        InferenceEngine::Precision,     // Net precision
        InferenceEngine::Precision,     // Input precision
        InferenceEngine::Precision,     // Output precision
        InferenceEngine::Layout,        // Input layout
        InferenceEngine::Layout,        // Output layout
        std::string                     // Target device name
> squeezeParams;

class SqueezeUnsqueezeLayerTest : public testing::WithParamInterface<squeezeParams>,
                       virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<squeezeParams>& obj);
protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
