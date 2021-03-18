// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <vector>
#include <string>
#include <memory>

#include "shared_test_classes/base/layer_test_utils.hpp"

#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

namespace LayerTestsDefinitions {

typedef std::tuple<
        ngraph::op::v4::Interpolate::InterpolateMode,          // InterpolateMode
        ngraph::op::v4::Interpolate::ShapeCalcMode,            // ShapeCalculationMode
        ngraph::op::v4::Interpolate::CoordinateTransformMode,  // CoordinateTransformMode
        ngraph::op::v4::Interpolate::NearestMode,              // NearestMode
        bool,                                                  // AntiAlias
        std::vector<size_t>,                                   // PadBegin
        std::vector<size_t>,                                   // PadEnd
        double,                                                // Cube coef
        std::vector<int64_t>,                                  // Axes
        std::vector<float>                                     // Scales
> InterpolateSpecificParams;

typedef std::tuple<
        InterpolateSpecificParams,
        InferenceEngine::Precision,     // Net precision
        InferenceEngine::Precision,     // Input precision
        InferenceEngine::Precision,     // Output precision
        InferenceEngine::Layout,        // Input layout
        InferenceEngine::Layout,        // Output layout
        InferenceEngine::SizeVector,    // Input shapes
        InferenceEngine::SizeVector,    // Target shapes
        LayerTestsUtils::TargetDevice   // Device name
> InterpolateLayerTestParams;

class InterpolateLayerTest : public testing::WithParamInterface<InterpolateLayerTestParams>,
                             virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<InterpolateLayerTestParams> obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
