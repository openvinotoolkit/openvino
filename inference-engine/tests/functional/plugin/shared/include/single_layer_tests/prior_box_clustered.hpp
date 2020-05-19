// Copyright (C) 2020 Intel Corporation
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
#include "details/ie_exception.hpp"

#include "ngraph/opsets/opset1.hpp"

#include "functional_test_utils/blob_utils.hpp"
#include "functional_test_utils/layer_test_utils.hpp"
#include "common_test_utils/common_utils.hpp"

#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"

typedef std::tuple<
    std::vector<float>,  // widths
    std::vector<float>,  // heights
    bool,                // clip
    float,               // step_width
    float,               // step_height
    float,               // offset
    std::vector<float>> priorBoxClusteredSpecificParams;

typedef std::tuple<
    priorBoxClusteredSpecificParams,
    InferenceEngine::Precision,   // net precision
    InferenceEngine::SizeVector,  // input shape
    InferenceEngine::SizeVector,  // image shape
    std::string> priorBoxClusteredLayerParams;

namespace LayerTestsDefinitions {
class PriorBoxClusteredLayerTest
    : public testing::WithParamInterface<priorBoxClusteredLayerParams>,
      public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<priorBoxClusteredLayerParams>& obj);

protected:
    InferenceEngine::SizeVector inputShapes;
    InferenceEngine::SizeVector imageShapes;
    InferenceEngine::Precision netPrecision;
    std::vector<float> widths;
    std::vector<float> heights;
    std::vector<float> variances;
    float step_width;
    float step_height;
    float offset;
    bool clip;

    std::vector<std::vector<std::uint8_t>> CalculateRefs() override;
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
