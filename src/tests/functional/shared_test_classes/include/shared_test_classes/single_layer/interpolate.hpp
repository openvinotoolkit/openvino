// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <vector>
#include <string>
#include <memory>

#include "shared_test_classes/base/layer_test_utils.hpp"

#include "ov_models/builders.hpp"
#include "ov_models/utils/ov_helpers.hpp"

namespace LayerTestsDefinitions {

typedef std::tuple<
        ov::op::util::InterpolateBase::InterpolateMode,          // InterpolateMode
        ov::op::util::InterpolateBase::ShapeCalcMode,            // ShapeCalculationMode
        ov::op::util::InterpolateBase::CoordinateTransformMode,  // CoordinateTransformMode
        ov::op::util::InterpolateBase::NearestMode,              // NearestMode
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
    static std::string getTestCaseName(const testing::TestParamInfo<InterpolateLayerTestParams>& obj);

protected:
    void SetUp() override;
};

namespace v11 {

class InterpolateLayerTest : public testing::WithParamInterface<InterpolateLayerTestParams>,
                             virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<InterpolateLayerTestParams>& obj);

protected:
    void SetUp() override;
};

} // namespace v11

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
