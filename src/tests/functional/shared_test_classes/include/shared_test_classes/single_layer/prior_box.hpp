// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <tuple>
#include <string>
#include <map>
#include <memory>
#include <set>
#include <functional>
#include <gtest/gtest.h>


#include "ie_core.hpp"
#include "ie_precision.hpp"

#include "ngraph/opsets/opset1.hpp"

#include "functional_test_utils/blob_utils.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "common_test_utils/common_utils.hpp"

#include "ov_models/utils/ov_helpers.hpp"
#include "ov_models/builders.hpp"

namespace LayerTestsDefinitions {
using priorBoxSpecificParams =  std::tuple<
        std::vector<float>, // min_size
        std::vector<float>, // max_size
        std::vector<float>, // aspect_ratio
        std::vector<float>, // density
        std::vector<float>, // fixed_ratio
        std::vector<float>, // fixed_size
        bool,               // clip
        bool,               // flip
        float,              // step
        float,              // offset
        std::vector<float>, // variance
        bool,               // scale_all_sizes
        bool>;              // min_max_aspect_ratios_order

typedef std::tuple<
        priorBoxSpecificParams,
        InferenceEngine::Precision,   // net precision
        InferenceEngine::Precision,   // Input precision
        InferenceEngine::Precision,   // Output precision
        InferenceEngine::Layout,      // Input layout
        InferenceEngine::Layout,      // Output layout
        InferenceEngine::SizeVector,  // input shape
        InferenceEngine::SizeVector,  // image shape
        std::string> priorBoxLayerParams;

class PriorBoxLayerTest
    : public testing::WithParamInterface<priorBoxLayerParams>,
      virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<priorBoxLayerParams>& obj);
protected:
    InferenceEngine::SizeVector inputShapes;
    InferenceEngine::SizeVector imageShapes;
    InferenceEngine::Precision netPrecision;
    std::vector<float> min_size;
    std::vector<float> max_size;
    std::vector<float> aspect_ratio;
    std::vector<float> density;
    std::vector<float> fixed_ratio;
    std::vector<float> fixed_size;
    std::vector<float> variance;
    float step;
    float offset;
    bool clip;
    bool flip;
    bool scale_all_sizes;
    bool min_max_aspect_ratios_order;

    void SetUp() override;
};

} // namespace LayerTestsDefinitions
