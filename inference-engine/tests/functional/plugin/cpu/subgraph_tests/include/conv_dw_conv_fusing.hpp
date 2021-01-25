// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <vector>
#include <string>

#include "test_utils/cpu_test_utils.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"

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
        size_t,                         // Number of groups
        size_t,                         // DW stride
        bool,                           // with bias
        bool                            // with dw bias
>;

using convDWConvFusingCPUParams = std::tuple<
        commonConvParams,                   // Convolution params
        CPUTestUtils::CPUSpecificParams,    // CPU runtime params
        InferenceEngine::SizeVector        // Input shapes
>;

class ConvDWConvFusingSubgraphTest : public testing::WithParamInterface<convDWConvFusingCPUParams>, public CPUTestsBase,
        virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<convDWConvFusingCPUParams> obj);

protected:
    void SetUp() override;
};

} // namespace SubgraphTestsDefinitions
