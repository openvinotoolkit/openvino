// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <vector>
#include <string>

#include "test_utils/cpu_test_utils.hpp"
#include "functional_test_utils/layer_test_utils.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"

using namespace CPUTestUtils;

namespace LayerTestsDefinitions {

enum class nodeType {
    convolution,
    convolutionBackpropData,
    groupConvolution,
    groupConvolutionBackpropData
};

std::string nodeType2PluginType(nodeType nt) {
    if (nt == nodeType::convolution) return "Convolution";
    if (nt == nodeType::convolutionBackpropData) return "Deconvolution";
    if (nt == nodeType::groupConvolution) return "Convolution";
    if (nt == nodeType::groupConvolutionBackpropData) return "Deconvolution";
    assert(!"unknown node type");
    return "undef";
}

std::string nodeType2str(nodeType nt) {
    if (nt == nodeType::convolution) return "Convolution";
    if (nt == nodeType::convolutionBackpropData) return "ConvolutionBackpropData";
    if (nt == nodeType::groupConvolution) return "GroupConvolution";
    if (nt == nodeType::groupConvolutionBackpropData) return "GroupConvolutionBackpropData";
    assert(!"unknown node type");
    return "undef";
}

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

} // namespace LayerTestsDefinitions