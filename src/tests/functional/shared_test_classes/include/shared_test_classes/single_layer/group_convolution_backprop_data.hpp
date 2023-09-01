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

// DEPRECATED, remove this old API when KMB (#58495) and ARM (#58496) plugins are migrated to new API
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
    static std::string getTestCaseName(const testing::TestParamInfo<groupConvBackpropDataLayerTestParamsSet>& obj);

protected:
    void SetUp() override;
};

using  groupConvBackpropSpecificParams = std::tuple<
    InferenceEngine::SizeVector,        // kernels
    InferenceEngine::SizeVector,        // strides
    std::vector<ptrdiff_t>,             // pad begins
    std::vector<ptrdiff_t>,             // pad ends
    InferenceEngine::SizeVector,        // dilations
    size_t,                             // num output channels
    size_t,                             // num groups
    ngraph::op::PadType,                // padding type
    std::vector<ptrdiff_t>>;            // output padding
using  groupConvBackpropLayerTestParamsSet = std::tuple<
        groupConvBackpropSpecificParams,
        InferenceEngine::Precision,     // Network precision
        InferenceEngine::Precision,     // Input precision
        InferenceEngine::Precision,     // Output precision
        InferenceEngine::Layout,        // Input layout
        InferenceEngine::Layout,        // Output layout
        InferenceEngine::SizeVector,    // Input shape
        InferenceEngine::SizeVector,    // Output shapes
        LayerTestsUtils::TargetDevice>; // Device name

class GroupConvBackpropLayerTest : public testing::WithParamInterface<groupConvBackpropLayerTestParamsSet>,
                                       virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<groupConvBackpropLayerTestParamsSet> obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
