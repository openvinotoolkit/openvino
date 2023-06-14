// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_layer/convolution.hpp"

#include "test_utils/cpu_test_utils.hpp"
#include "test_utils/convolution_params.hpp"
#include "test_utils/fusing_test_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"
#include "openvino/core/visibility.hpp"
#include <shared_test_classes/single_layer/convolution.hpp>

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ov::test;

namespace CPULayerTestsDefinitions {
using LayerTestsDefinitions::convSpecificParams;

typedef std::tuple<
        convSpecificParams,
        ElementType,     // Net precision
        ElementType,     // Input precision
        ElementType,     // Output precision
        InputShape,      // Input shape
        LayerTestsUtils::TargetDevice   // Device name
> convLayerTestParamsSet;

typedef std::tuple<
        convLayerTestParamsSet,
        CPUSpecificParams,
        fusingSpecificParams,
        std::map<std::string, std::string> > convLayerCPUTestParamsSet;

class ConvolutionLayerCPUTest : public testing::WithParamInterface<convLayerCPUTestParamsSet>,
                                virtual public SubgraphBaseTest, public CpuTestWithFusing {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<convLayerCPUTestParamsSet>& obj);
protected:
    bool isBias = false;
    InferenceEngine::SizeVector kernel, dilation;

    void checkBiasFusing(ov::CompiledModel &execNet) const;
    std::shared_ptr<ngraph::Node> modifyGraph(const ngraph::element::Type &ngPrc,
                                              ngraph::ParameterVector &params,
                                              const std::shared_ptr<ngraph::Node> &lastNode) override;
    void SetUp() override;
};

namespace Convolution {
} // namespace Convolution
} // namespace CPULayerTestsDefinitions