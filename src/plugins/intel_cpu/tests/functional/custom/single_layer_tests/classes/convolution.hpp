// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "common_test_utils/node_builders/convolution.hpp"
#include "openvino/core/visibility.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "shared_test_classes/single_op/convolution.hpp"
#include "utils/convolution_params.hpp"
#include "utils/cpu_test_utils.hpp"
#include "utils/fusing_test_utils.hpp"
#include "utils/quantization_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {
namespace Convolution {

typedef std::tuple<
    convSpecificParams,
    ElementType,     // Net precision
    ElementType,     // Input precision
    ElementType,     // Output precision
    InputShape,      // Input shape
    ov::test::TargetDevice   // Device name
    > convLayerTestParamsSet;

typedef std::tuple<
    convLayerTestParamsSet,
    CPUSpecificParams,
    ExtraOperationsParams,
    ov::AnyMap
    > convLayerCPUTestParamsSet;

class ConvolutionLayerCPUTest : public testing::WithParamInterface<convLayerCPUTestParamsSet>,
                                virtual public SubgraphBaseTest, public CpuTestWithFusing {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<convLayerCPUTestParamsSet>& obj);
protected:
    bool isBias = false;
    ov::Shape kernel, dilation;
    ov::Shape stride;
    std::vector<ptrdiff_t> padBegin, padEnd;

    void checkBiasFusing(ov::CompiledModel &execNet) const;
    std::shared_ptr<ov::Node> modifyGraph(const ov::element::Type &ngPrc,
                                              ov::ParameterVector &params,
                                              const std::shared_ptr<ov::Node> &lastNode) override;
    void SetUp() override;
    void generate_inputs(const std::vector<ov::Shape> &targetInputStaticShapes) override;
};

    using SizeVector = std::vector<size_t>;
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

    const std::vector<CPUSpecificParams>& CPUParams_1x1_1D();
    const std::vector<CPUSpecificParams>& CPUParams_1x1_2D();
    const std::vector<CPUSpecificParams>& CPUParams_2D();
    const std::vector<CPUSpecificParams>& CPUParams_3D();
    const std::vector<CPUSpecificParams>& CPUParams_GEMM_1D();
    const std::vector<CPUSpecificParams>& CPUParams_GEMM_2D();
    const std::vector<CPUSpecificParams>& CPUParams_GEMM_3D();

    const std::vector<InputShape>& inputShapes1d();
    const std::vector<InputShape>& inputShapes2d();
    const std::vector<InputShape>& inputShapes3d();
    const std::vector<InputShape>& inputShapes2d_cache();
    const std::vector<InputShape>& inputShapesPlain2Blocked2d();
    const std::vector<InputShape>& inputShapes2d_dynBatch();
    const std::vector<InputShape>& inShapesGemm1D();

    const std::vector<InputShape>& inShapesGemm2D();
    const std::vector<InputShape>& inShapesGemm2D_cache();
    const std::vector<InputShape>& inShapesGemm3D();

    const ov::Shape& numOutChannels();
    const ov::Shape& numOutChannels_Gemm();

    const std::vector<ExtraOperationsParams>& fusingParamsSetWithEmpty();

    using convParams_ExplicitPaddingType = decltype(::testing::Combine(
                                                        ::testing::ValuesIn(kernels2d()),
                                                        ::testing::ValuesIn(strides2d()),
                                                        ::testing::ValuesIn(padBegins2d()),
                                                        ::testing::ValuesIn(padEnds2d()),
                                                        ::testing::ValuesIn(dilations2d()),
                                                        ::testing::ValuesIn(numOutChannels_Gemm()),
                                                        ::testing::Values(ov::op::PadType::EXPLICIT)));
    using convParams_ExplicitPaddingDilatedType = decltype(::testing::Combine(
                                                                ::testing::ValuesIn(kernels2d()),
                                                                ::testing::ValuesIn(strides2d()),
                                                                ::testing::ValuesIn(padBegins2d()),
                                                                ::testing::ValuesIn(padEnds2d()),
                                                                ::testing::Values(ov::Shape{2, 2}),
                                                                ::testing::ValuesIn(numOutChannels_Gemm()),
                                                                ::testing::Values(ov::op::PadType::EXPLICIT)));
    using convParams_ExplicitPadding_1x1_Type = decltype(::testing::Combine(
                                                                ::testing::Values(ov::Shape({1})),
                                                                ::testing::Values(ov::Shape({1})),
                                                                ::testing::Values(std::vector<ptrdiff_t>({0})),
                                                                ::testing::Values(std::vector<ptrdiff_t>({0})),
                                                                ::testing::Values(ov::Shape({1})),
                                                                ::testing::Values(63),
                                                                ::testing::Values(ov::op::PadType::EXPLICIT)));
    const convParams_ExplicitPaddingType& convParams_ExplicitPadding_GEMM_1D();
    const convParams_ExplicitPaddingType& convParams_ExplicitPadding_GEMM_2D();
    const convParams_ExplicitPaddingType& convParams_ExplicitPadding_GEMM_3D();
    const convParams_ExplicitPaddingType& convParams_ExplicitPadding_2D();
    const convParams_ExplicitPaddingType& convParams_ExplicitPadding_3D();

    const convParams_ExplicitPaddingDilatedType& convParams_ExplicitPadding_2D_dilated();
    const convParams_ExplicitPaddingDilatedType& convParams_ExplicitPadding_3D_dilated();
    const convParams_ExplicitPaddingDilatedType& convParams_ExplicitPadding_GEMM_2D_dilated();
    const convParams_ExplicitPaddingDilatedType& convParams_ExplicitPadding_GEMM_3D_dilated();

    const convParams_ExplicitPadding_1x1_Type& convParams_ExplicitPadding_1x1_1D();
    const convParams_ExplicitPadding_1x1_Type& convParams_ExplicitPadding_1x1_2D();
    }  // namespace Convolution
    }  // namespace test
    }  // namespace ov
