// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <vector>
#include <string>

#include "test_utils/cpu_test_utils.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ov_models/utils/ov_helpers.hpp"
#include "ov_models/builders.hpp"

using namespace CPUTestUtils;

namespace SubgraphTestsDefinitions {

using commonConvParams =  std::tuple<
    InferenceEngine::SizeVector,    // Kernel size
    InferenceEngine::SizeVector,    // Strides
    std::vector<ptrdiff_t>,         // Pad begin
    std::vector<ptrdiff_t>,         // Pad end
    InferenceEngine::SizeVector,    // Dilation
    size_t,                         // Num out channels
    ngraph::op::PadType,            // Padding type
    size_t                          // Number of groups
>;

using convConcatCPUParams = std::tuple<
    nodeType,                           // Ngraph convolution type
    commonConvParams,                   // Convolution params
    CPUTestUtils::CPUSpecificParams,    // CPU runtime params
    InferenceEngine::SizeVector,        // Input shapes
    int                                 // Axis for concat
>;

class ConvConcatSubgraphTest : public testing::WithParamInterface<convConcatCPUParams>, public CPUTestsBase, virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<convConcatCPUParams> obj);

protected:
    void SetUp() override;
    std::string pluginTypeNode;
};

} // namespace SubgraphTestsDefinitions
