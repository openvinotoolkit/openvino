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
    InferenceEngine::SizeVector stride;
    std::vector<ptrdiff_t> padBegin, padEnd;

    void checkBiasFusing(ov::CompiledModel &execNet) const;
    std::shared_ptr<ngraph::Node> modifyGraph(const ngraph::element::Type &ngPrc,
                                              ngraph::ParameterVector &params,
                                              const std::shared_ptr<ngraph::Node> &lastNode) override;
    void SetUp() override;
};

namespace Convolution {
    const std::vector<SizeVector>& kernels1d();
    const std::vector<SizeVector>& strides1d();
    const std::vector<std::vector<ptrdiff_t>>& padBegins1d();
    const std::vector<std::vector<ptrdiff_t>>& padEnds1d();
    const std::vector<SizeVector>& dilations1d();

    const std::vector<SizeVector>& kernels2d();
    const std::vector<SizeVector>& strides2d();
    const std::vector<std::vector<ptrdiff_t>>& padBegins2d();
    const std::vector<std::vector<ptrdiff_t>>& padEnds2d();
    const std::vector<SizeVector>& dilations2d();

    const std::vector<SizeVector>& kernels3d();
    const std::vector<SizeVector>& strides3d();
    const std::vector<std::vector<ptrdiff_t>>& padBegins3d();
    const std::vector<std::vector<ptrdiff_t>>& padEnds3d();
    const std::vector<SizeVector>& dilations3d();
    const std::vector<InputShape> & inputShapes3d();

    const std::vector<CPUSpecificParams>& CPUParams_1x1_1D();
    const std::vector<CPUSpecificParams>& CPUParams_1x1_2D();
    const std::vector<CPUSpecificParams>& CPUParams_2D();
    const std::vector<CPUSpecificParams>& CPUParams_GEMM_2D();
    const std::vector<CPUSpecificParams>& CPUParams_GEMM_3D();

    const std::vector<InputShape>& inputShapes1d();
    const std::vector<InputShape>& inputShapes2d();
    const std::vector<InputShape>& inputShapes2d_cache();
    const std::vector<InputShape>& inputShapesPlain2Blocked2d();
    const std::vector<InputShape>& inputShapes2d_dynBatch();

    const std::vector<InputShape>& inShapesGemm2D();
    const std::vector<InputShape>& inShapesGemm2D_cache();
    const std::vector<InputShape> & inShapesGemm3D();

    const SizeVector& numOutChannels();
    const SizeVector& numOutChannels_Gemm();
} // namespace Convolution
} // namespace CPULayerTestsDefinitions