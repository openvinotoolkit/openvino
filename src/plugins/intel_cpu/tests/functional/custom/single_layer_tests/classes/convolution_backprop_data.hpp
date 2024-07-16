// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/convolution_backprop_data.hpp"
#include "common_test_utils/node_builders/convolution_backprop_data.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "cpu_shape.h"
#include "openvino/core/preprocess/pre_post_process.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/convolution_params.hpp"
#include "utils/cpu_test_utils.hpp"
#include "utils/filter_cpu_info.hpp"
#include "utils/fusing_test_utils.hpp"
#include <openvino/opsets/opset8.hpp>

using namespace CPUTestUtils;
namespace ov {
namespace test {

using DeconvSpecParams = ov::test::convBackpropDataSpecificParams;

using DeconvInputData = std::tuple<InputShape,                          // data shape
        ov::test::utils::InputLayerType,     // 'output_shape' input type
        std::vector<std::vector<int32_t>>>;  // values for 'output_shape'

using DeconvLayerCPUTestParamsSet =
        std::tuple<DeconvSpecParams, DeconvInputData, ElementType, fusingSpecificParams, CPUSpecificParams, ov::AnyMap>;

class DeconvolutionLayerCPUTest : public testing::WithParamInterface<DeconvLayerCPUTestParamsSet>,
                                  virtual public SubgraphBaseTest,
                                  public CpuTestWithFusing {
public:
static std::string getTestCaseName(testing::TestParamInfo<DeconvLayerCPUTestParamsSet> obj);

void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;

void configure_model() override;

std::shared_ptr<ov::Model> createGraph(const std::vector<ov::PartialShape>& inShapes,
                                       ov::test::utils::InputLayerType outShapeType);

protected:
std::vector<size_t> kernel, stride;

void SetUp() override;

private:
    ElementType prec;
    ov::op::PadType padType;
    std::vector<size_t> dilation;
    std::vector<ptrdiff_t> padBegin, padEnd, outPadding;
    size_t convOutChannels;
    std::vector<std::vector<int32_t>> outShapeData;
    size_t inferRequestNum = 0;
};

/* COMMON PARAMS */
const std::vector<fusingSpecificParams> fusingParamsSet{
        emptyFusingSpec,
#if !defined(OPENVINO_ARCH_ARM64) && !defined(OPENVINO_ARCH_ARM)
        fusingAddPerChannel
#endif
};


/* COMMON PARAMS */
const std::vector<std::vector<ptrdiff_t>> emptyOutputPadding = {{}};

/* ============= Deconvolution params (planar layout) ============= */
const std::vector<size_t> numOutChannels_Planar = {6};

/* ============= Deconvolution params (blocked layout) ============= */
const std::vector<size_t> numOutChannels_Blocked = {64};

/* ============= Deconvolution params (2D) ============= */
const std::vector<std::vector<size_t>> kernels2d = {{3, 3}, {1, 1}};
const std::vector<std::vector<size_t>> strides2d = {{1, 1}, {2, 2}};
const std::vector<std::vector<ptrdiff_t>> padBegins2d = {{0, 0}};
const std::vector<std::vector<ptrdiff_t>> padEnds2d = {{0, 0}};
const std::vector<std::vector<size_t>> dilations2d = {{1, 1}};

const std::vector<std::vector<size_t>> deconvBrgKernels2d = {{3, 3}, {2, 2}};
const std::vector<std::vector<size_t>> deconvBrgKernels2d_1x1 = {{1, 1}};
const std::vector<std::vector<size_t>> deconvBrgStrides2d = {{1, 1}};

/* ============= Deconvolution params (3D) ============= */
const std::vector<std::vector<size_t>> kernels3d = {{3, 3, 3}, {1, 1, 1}};
const std::vector<std::vector<size_t>> strides3d = {{1, 1, 1}, {2, 2, 2}};
const std::vector<std::vector<ptrdiff_t>> padBegins3d = {{0, 0, 0}};
const std::vector<std::vector<ptrdiff_t>> padEnds3d = {{0, 0, 0}};
const std::vector<std::vector<size_t>> dilations3d = {{1, 1, 1}};

const std::vector<std::vector<size_t>> deconvBrgKernels3d = {{3, 3, 3}, {2, 2, 2}};
const std::vector<std::vector<size_t>> deconvBrgKernels3d_1x1 = {{1, 1, 1}};
const std::vector<std::vector<size_t>> deconvBrgStrides3d = {{1, 1, 1}};

/* ============= */

} // namespace test
} // namespace ov
