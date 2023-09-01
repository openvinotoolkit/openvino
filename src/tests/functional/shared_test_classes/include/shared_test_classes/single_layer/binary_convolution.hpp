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

using binConvSpecificParams = std::tuple<
    InferenceEngine::SizeVector,    // Kernel size
    InferenceEngine::SizeVector,    // Strides
    std::vector<ptrdiff_t>,         // Pads begin
    std::vector<ptrdiff_t>,         // Pads end
    InferenceEngine::SizeVector,    // Dilations
    size_t,                         // Num Output channels
    ngraph::op::PadType,            // Padding type
    float>;                         // Padding value

using binaryConvolutionTestParamsSet = std::tuple<
    binConvSpecificParams,          //
    InferenceEngine::Precision,     // Network precision
    InferenceEngine::Precision,     // Input precision
    InferenceEngine::Precision,     // Output precision
    InferenceEngine::Layout,        // Input layout
    InferenceEngine::Layout,        // Output layout
    InferenceEngine::SizeVector,    // Input shape
    LayerTestsUtils::TargetDevice>; // Device name

class BinaryConvolutionLayerTest : public testing::WithParamInterface<binaryConvolutionTestParamsSet>,
                                   virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<binaryConvolutionTestParamsSet>& obj);
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo &info) const override;

protected:
    void SetUp() override;
};

} // namespace LayerTestsDefinitions
