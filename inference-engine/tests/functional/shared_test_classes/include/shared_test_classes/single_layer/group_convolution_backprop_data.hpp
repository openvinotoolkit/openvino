// Copyright (C) 2021 Intel Corporation
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

using  groupConvBackpropDataSpecificParams = std::tuple<
    InferenceEngine::SizeVector,        // kernels
    InferenceEngine::SizeVector,        // strides
    std::vector<ptrdiff_t>,             // pad begins
    std::vector<ptrdiff_t>,             // pad ends
    InferenceEngine::SizeVector,        // dilations
    size_t,                             // num output channels
    size_t,                             // num groups
    ngraph::op::PadType>;               // padding type
using  groupConvBackpropDataLayerTestParamsSet = std::tuple<
        groupConvBackpropDataSpecificParams,
        InferenceEngine::Precision,     // Network precision
        InferenceEngine::Precision,     // Input precision
        InferenceEngine::Precision,     // Output precision
        InferenceEngine::Layout,        // Input layout
        InferenceEngine::Layout,        // Output layout
        InferenceEngine::SizeVector,    // Input shape
        LayerTestsUtils::TargetDevice>; // Device name

class GroupConvBackpropDataLayerTest : public testing::WithParamInterface<groupConvBackpropDataLayerTestParamsSet>,
                                       virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<groupConvBackpropDataLayerTestParamsSet> obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions