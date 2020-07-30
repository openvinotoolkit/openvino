// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <vector>
#include <string>
#include <memory>

#include "functional_test_utils/layer_test_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

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
        InferenceEngine::SizeVector,    // Input shapes
        LayerTestsUtils::TargetDevice   // Device name
> convLayerTestParamsSet;
namespace LayerTestsDefinitions {


class ConvolutionLayerTest : public testing::WithParamInterface<convLayerTestParamsSet>,
                             public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<convLayerTestParamsSet> obj);

protected:
    void SetUp() override;
};
// ! [test_convolution:definition]

}  // namespace LayerTestsDefinitions
