// Copyright (C) 2020 Intel Corporation
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

typedef std::tuple<
        InferenceEngine::SizeVector,    // Kernel size
        InferenceEngine::SizeVector,    // Strides
        std::vector<ptrdiff_t>,         // Pad begin
        std::vector<ptrdiff_t>,         // Pad end
        InferenceEngine::SizeVector,    // Dilation
        size_t,                         // Num out channels
        ngraph::op::PadType             // Padding type
> convBackpropDataSpecificParams;
typedef std::tuple<
        convBackpropDataSpecificParams,
        InferenceEngine::Precision,     // Net precision
        InferenceEngine::Precision,     // Input precision
        InferenceEngine::Precision,     // Output precision
        InferenceEngine::Layout,        // Input layout
        InferenceEngine::Layout,        // Output layout
        InferenceEngine::SizeVector,    // Input shapes
        LayerTestsUtils::TargetDevice   // Device name
> convBackpropDataLayerTestParamsSet;
namespace LayerTestsDefinitions {


class ConvolutionBackpropDataLayerTest : public testing::WithParamInterface<convBackpropDataLayerTestParamsSet>,
                                         virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<convBackpropDataLayerTestParamsSet> obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
