// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <memory>

#include "ov_models/builders.hpp"
#include "ov_models/utils/ov_helpers.hpp"

#include "shared_test_classes/base/layer_test_utils.hpp"

namespace LayerTestsDefinitions {

using roiPoolingParamsTuple = std::tuple<
        InferenceEngine::SizeVector,                // Input shape
        InferenceEngine::SizeVector,                // Coords shape
        std::vector<size_t>,                        // Pooled shape {pooled_h, pooled_w}
        float,                                      // Spatial scale
        ngraph::helpers::ROIPoolingTypes,           // ROIPooling method
        InferenceEngine::Precision,                 // Net precision
        LayerTestsUtils::TargetDevice>;             // Device name

class ROIPoolingLayerTest : public testing::WithParamInterface<roiPoolingParamsTuple>,
                            virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<roiPoolingParamsTuple>& obj);
    void GenerateInputs() override;

protected:
    void SetUp() override;

private:
    ngraph::helpers::ROIPoolingTypes pool_method;
    float spatial_scale;
};

}  // namespace LayerTestsDefinitions
