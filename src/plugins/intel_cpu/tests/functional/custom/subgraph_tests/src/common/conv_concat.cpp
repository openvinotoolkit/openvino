// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/subgraph_tests/src/classes/conv_concat.hpp"

#include "utils/convolution_params.hpp"
#include "utils/filter_cpu_info.hpp"

using namespace CPUTestUtils;
using namespace ov::test::ConvConcat;

namespace ov {
namespace test {
namespace ConvolutionBackpropDataConcat {

/* ============= ConvolutionBackpropData (2D) ============= */
const std::vector<CPUSpecificParams> CPUParams2D = {
    planar_2D
};

const auto params2D = ::testing::Combine(
    ::testing::Values(nodeType::convolutionBackpropData),
    ::testing::Values(convParams2D()),
    ::testing::ValuesIn(filterCPUInfo(CPUParams2D)),
    ::testing::Values(inputShapes2D()),
    ::testing::Values(axis())
);

INSTANTIATE_TEST_SUITE_P(smoke_ConvolutionBackpropData2D, ConvConcatSubgraphTest, params2D, ConvConcatSubgraphTest::getTestCaseName);

/* ============= ConvolutionBackpropData (3D) ============= */
const std::vector<CPUSpecificParams> CPUParams3D = {
    planar_3D
};

const auto params3D = ::testing::Combine(
    ::testing::Values(nodeType::convolutionBackpropData),
    ::testing::Values(convParams3D()),
    ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams3D)),
    ::testing::Values(inputShapes3D()),
    ::testing::Values(axis())
);

INSTANTIATE_TEST_SUITE_P(smoke_ConvolutionBackpropData3D, ConvConcatSubgraphTest, params3D, ConvConcatSubgraphTest::getTestCaseName);

}  // namespace ConvolutionBackpropDataConcat

namespace ConvolutionConact {

/* ============= Convolution (2D) ============= */
const std::vector<CPUSpecificParams> CPUParams2D = {
    conv_gemm_2D
};

const auto params2D = ::testing::Combine(
    ::testing::Values(nodeType::convolution),
    ::testing::Values(convParams2D()),
    ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams2D)),
    ::testing::Values(inputShapes2D()),
    ::testing::Values(axis())
);

INSTANTIATE_TEST_SUITE_P(smoke_Convolution2D, ConvConcatSubgraphTest, params2D, ConvConcatSubgraphTest::getTestCaseName);

/* ============= Convolution (3D) ============= */
const std::vector<CPUSpecificParams> CPUParams3D = {
    conv_gemm_3D
};

const auto params3D = ::testing::Combine(
    ::testing::Values(nodeType::convolution),
    ::testing::Values(convParams3D()),
    ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams3D)),
    ::testing::Values(inputShapes3D()),
    ::testing::Values(axis())
);

INSTANTIATE_TEST_SUITE_P(smoke_Convolution3D, ConvConcatSubgraphTest, params3D, ConvConcatSubgraphTest::getTestCaseName);

}  // namespace ConvolutionConact

namespace GroupConvolutionConcat {

/* ============= GroupConvolution (2D) ============= */
const std::vector<CPUSpecificParams> CPUParams2D = {
    conv_gemm_2D
};

const auto params2D = ::testing::Combine(
    ::testing::Values(nodeType::groupConvolution),
    ::testing::Values(groupConvParams2D()),
    ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams2D)),
    ::testing::Values(inputShapes2D()),
    ::testing::Values(axis())
);

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvolution2D, ConvConcatSubgraphTest, params2D, ConvConcatSubgraphTest::getTestCaseName);

/* ============= GroupConvolution (3D) ============= */
const std::vector<CPUSpecificParams> CPUParams3D = {
    conv_gemm_3D
};

const auto params3D = ::testing::Combine(
    ::testing::Values(nodeType::groupConvolution),
    ::testing::Values(groupConvParams3D()),
    ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams3D)),
    ::testing::Values(inputShapes3D()),
    ::testing::Values(axis())
);

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvolution3D, ConvConcatSubgraphTest, params3D, ConvConcatSubgraphTest::getTestCaseName);

}  // namespace GroupConvolutionConcat

namespace GroupConvolutionBackpropDataConcat {

/* ============= GroupConvolutionBackpropData (2D) ============= */
const std::vector<CPUSpecificParams> CPUParams2D = {
    planar_2D
};

const auto params2D = ::testing::Combine(
    ::testing::Values(nodeType::groupConvolutionBackpropData),
    ::testing::Values(groupConvParams2D()),
    ::testing::ValuesIn(filterCPUSpecificParams(CPUParams2D)),
    ::testing::Values(inputShapes2D()),
    ::testing::Values(axis())
);

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvolutionBackpropData2D, ConvConcatSubgraphTest, params2D, ConvConcatSubgraphTest::getTestCaseName);

/* ============= GroupConvolutionBackpropData (3D) ============= */
const std::vector<CPUSpecificParams> CPUParams3D = {
    planar_3D
};

const auto params3D = ::testing::Combine(
    ::testing::Values(nodeType::groupConvolutionBackpropData),
    ::testing::Values(groupConvParams3D()),
    ::testing::ValuesIn(filterCPUSpecificParams(CPUParams3D)),
    ::testing::Values(inputShapes3D()),
    ::testing::Values(axis())
);

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvolutionBackpropData3D, ConvConcatSubgraphTest, params3D, ConvConcatSubgraphTest::getTestCaseName);

}  // namespace GroupConvolutionBackpropDataConcat

}  // namespace test
}  // namespace ov
