// Copyright (C) 2018-2021 Intel Corporation
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
        InferenceEngine::Precision,        // Net precision
        InferenceEngine::Precision,        // Input precision
        InferenceEngine::Precision,        // Output precision
        InferenceEngine::Layout,           // Input layout
        InferenceEngine::Layout,           // Output layout
        InferenceEngine::SizeVector,       // Input shapes
        InferenceEngine::SizeVector,       // Target shapes
        LayerTestsUtils::TargetDevice,     // Device name
        std::map<std::string, std::string> // Additional network configuration
> InterpolateLayerTestParams;

class InterpolateLayerTest : public testing::WithParamInterface<InterpolateLayerTestParams>,
                             virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<InterpolateLayerTestParams> obj);

protected:
    void SetUp() override;
};

//Interpolate-1 test
typedef std::tuple<
        InferenceEngine::Precision,        // Net precision
        InferenceEngine::Precision,        // Input precision, output is the same
        InferenceEngine::Layout,           // Input layout, output is the same
        InferenceEngine::SizeVector,       // Input shapes
        InferenceEngine::SizeVector,       // Target shapes
        std::string,                       // InterpolateMode
        ngraph::AxisSet,                   // Axes
        bool,                              // AntiAlias
        std::vector<size_t>,               // Pads
        LayerTestsUtils::TargetDevice      // Device name
> Interpolate1LayerTestParams;

class Interpolate1LayerTest : public testing::WithParamInterface<Interpolate1LayerTestParams>,
                             virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<Interpolate1LayerTestParams> obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
