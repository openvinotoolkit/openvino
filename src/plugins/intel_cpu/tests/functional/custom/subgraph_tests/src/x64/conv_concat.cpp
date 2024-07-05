// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/subgraph_tests/include/conv_concat.hpp"

#include "common_test_utils/node_builders/convolution.hpp"
#include "common_test_utils/node_builders/convolution_backprop_data.hpp"
#include "common_test_utils/node_builders/group_convolution.hpp"
#include "common_test_utils/node_builders/group_convolution_backprop_data.hpp"
#include "utils/convolution_params.hpp"
#include "utils/filter_cpu_info.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {
namespace ConvolutionConcat {
/* ============= Convolution (2D) ============= */
const std::vector<CPUSpecificParams> CPUParams2D = {
    conv_ref_2D
};

const auto params2D = ::testing::Combine(
    ::testing::Values(nodeType::convolution),
    ::testing::Values(convParams2D),
    ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams2D)),
    ::testing::Values(inputShapes2D),
    ::testing::Values(axis)
);

INSTANTIATE_TEST_SUITE_P(smoke_Convolution2D, ConvConcatSubgraphTest, params2D, ConvConcatSubgraphTest::getTestCaseName);

/* ============= Convolution (3D) ============= */
const std::vector<CPUSpecificParams> CPUParams3D = {
    conv_ref_3D,
    conv_gemm_3D,
    conv_avx2_3D,
    conv_avx512_3D
};

const auto params3D = ::testing::Combine(
    ::testing::Values(nodeType::convolution),
    ::testing::Values(convParams3D),
    ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams3D)),
    ::testing::Values(inputShapes3D),
    ::testing::Values(axis)
);

INSTANTIATE_TEST_SUITE_P(smoke_Convolution3D, ConvConcatSubgraphTest, params3D, ConvConcatSubgraphTest::getTestCaseName);
} // namespace ConvolutionConcat

namespace GroupConvolutionConcat {
/* ============= GroupConvolution (2D) ============= */
const std::vector<CPUSpecificParams> CPUParams2D = {
    conv_ref_2D
};

const auto params2D = ::testing::Combine(
    ::testing::Values(nodeType::groupConvolution),
    ::testing::Values(groupConvParams2D),
    ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams2D)),
    ::testing::Values(inputShapes2D),
    ::testing::Values(axis)
);

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvolution2D, ConvConcatSubgraphTest, params2D, ConvConcatSubgraphTest::getTestCaseName);

/* ============= GroupConvolution (3D) ============= */
const std::vector<CPUSpecificParams> CPUParams3D = {
    conv_ref_3D,
    conv_gemm_3D,
    conv_avx2_3D,
    conv_avx512_3D
};

const auto params3D = ::testing::Combine(
    ::testing::Values(nodeType::groupConvolution),
    ::testing::Values(groupConvParams3D),
    ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams3D)),
    ::testing::Values(inputShapes3D),
    ::testing::Values(axis)
);

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvolution3D, ConvConcatSubgraphTest, params3D, ConvConcatSubgraphTest::getTestCaseName);

} // namespace GroupConvolutionConcat
}  // namespace test
}  // namespace ov