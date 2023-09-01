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

// ! [test_convolution:definition]
typedef std::tuple<
        InferenceEngine::SizeVector,    // Kernel size
        InferenceEngine::SizeVector,    // Strides
        std::vector<ptrdiff_t>,         // Pad begin
        std::vector<ptrdiff_t>,         // Pad end
        InferenceEngine::SizeVector,    // Dilation
        size_t,                         // Num out channels
        ngraph::op::PadType             // Padding type
> convSpecificParams;
typedef std::tuple<
        convSpecificParams,
        InferenceEngine::Precision,     // Net precision
        InferenceEngine::Precision,     // Input precision
        InferenceEngine::Precision,     // Output precision
        InferenceEngine::Layout,        // Input layout
        InferenceEngine::Layout,        // Output layout
        InferenceEngine::SizeVector,    // Input shapes
        LayerTestsUtils::TargetDevice   // Device name
> convLayerTestParamsSet;

class ConvolutionLayerTest : public testing::WithParamInterface<convLayerTestParamsSet>,
                             virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<convLayerTestParamsSet>& obj);

protected:
    void SetUp() override;
};
// ! [test_convolution:definition]

}  // namespace LayerTestsDefinitions
