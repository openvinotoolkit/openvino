// Copyright (C) 2020 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <memory>

#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

#include "functional_test_utils/layer_test_utils.hpp"

namespace LayerTestsDefinitions {
typedef std::tuple<
        ngraph::helpers::ROIPoolingTypes,  // ROIPooling type, max or bilinear
        std::vector<size_t>,               // Shape
        float                              // Scale
> roiPoolingSpecificParams;

typedef std::tuple<
        roiPoolingSpecificParams,
        InferenceEngine::Precision,     // Net precision
        InferenceEngine::Precision,     // Input precision
        InferenceEngine::Precision,     // Output precision
        InferenceEngine::Layout,        // Input layout
        InferenceEngine::Layout,        // Output layout
        InferenceEngine::SizeVector,    // Input shape
        InferenceEngine::SizeVector,    // Coords shape
        LayerTestsUtils::TargetDevice   // Device name
> roiPoolingLayerTestParamsSet;


class ROIPoolingLayerTest : public testing::WithParamInterface<roiPoolingLayerTestParamsSet>,
                            virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<roiPoolingLayerTestParamsSet> obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
