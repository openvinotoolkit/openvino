// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/subgraph_tests/src/classes/conv_concat.hpp"

#include "common_test_utils/node_builders/convolution.hpp"
#include "common_test_utils/node_builders/convolution_backprop_data.hpp"
#include "common_test_utils/node_builders/group_convolution.hpp"
#include "common_test_utils/node_builders/group_convolution_backprop_data.hpp"
#include "utils/convolution_params.hpp"
#include "utils/filter_cpu_info.hpp"

using namespace CPUTestUtils;
using namespace ov::test::ConvConcat;

namespace ov {
namespace test {
namespace Kernel_1x1 {

/* ============= Kernel_1x1 (2D) ============= */
const std::vector<CPUSpecificParams> CPUParams2DConv = {
    conv_avx2_2D_1x1,
    conv_avx512_2D_1x1
};

commonConvParams convParams2D1x1 = commonConvParams{{1, 1}, {1, 1}, {0, 0}, {0, 0}, dilation2D(), numOutChannels(), paddingType(), 1};

const auto params2DConv = ::testing::Combine(
    ::testing::Values(nodeType::convolution),
    ::testing::Values(convParams2D1x1),
    ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams2DConv)),
    ::testing::Values(inputShapes2D()),
    ::testing::Values(axis())
);

INSTANTIATE_TEST_SUITE_P(smoke_Convolution2D1x1, ConvConcatSubgraphTest, params2DConv, ConvConcatSubgraphTest::getTestCaseName);

const std::vector<CPUSpecificParams> CPUParams2DDeconv = {
    block8c_2D,
    block16c_2D
};

const auto params2DDeconv = ::testing::Combine(
    ::testing::Values(nodeType::convolutionBackpropData),
    ::testing::Values(convParams2D1x1),
    ::testing::ValuesIn(filterCPUInfo(CPUParams2DDeconv)),
    ::testing::Values(inputShapes2D()),
    ::testing::Values(axis())
);

INSTANTIATE_TEST_SUITE_P(smoke_ConvolutionBackpropData2D1x1, ConvConcatSubgraphTest, params2DDeconv, ConvConcatSubgraphTest::getTestCaseName);

}  // namespace Kernel_1x1

namespace GroupConvolutionBackpropDataDWConcat {

/* ============= GroupConvolutionBackpropData (DW 2D) ============= */
commonConvParams dwDeconvParams2D = commonConvParams{kernelSize2D(), strides2D(), padBegin2D(), padEnd2D(), dilation2D(),
                                                     numOutChannels(), paddingType(), numOutChannels()};
const ov::Shape inputShapesDW2D{1, 32, 16, 16};
const std::vector<CPUSpecificParams> CPUParams2D = {
    block8c_2D,
    block16c_2D
};

const auto params2D = ::testing::Combine(
    ::testing::Values(nodeType::groupConvolutionBackpropData),
    ::testing::Values(dwDeconvParams2D),
    ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams2D)),
    ::testing::Values(inputShapesDW2D),
    ::testing::Values(axis())
);

INSTANTIATE_TEST_SUITE_P(smoke_DWGroupConvolutionBackpropData2D, ConvConcatSubgraphTest, params2D, ConvConcatSubgraphTest::getTestCaseName);

}  // namespace GroupConvolutionBackpropDataDWConcat

namespace GroupConvolutionDWConcat {

/* ============= GroupConvolution (DW 2D) ============= */
commonConvParams dwConvParams2D = commonConvParams{kernelSize2D(), strides2D(), padBegin2D(), padEnd2D(), dilation2D(),
                                                   numOutChannels(), paddingType(), numOutChannels()};
const ov::Shape inputShapesDW2D{1, 32, 16, 16};
const std::vector<CPUSpecificParams> CPUParams2D = {
    conv_avx2_dw_2D,
    conv_avx512_dw_2D
};

const auto params2D = ::testing::Combine(
    ::testing::Values(nodeType::groupConvolution),
    ::testing::Values(dwConvParams2D),
    ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams2D)),
    ::testing::Values(inputShapesDW2D),
    ::testing::Values(axis())
);

INSTANTIATE_TEST_SUITE_P(smoke_DWGroupConvolution2D, ConvConcatSubgraphTest, params2D, ConvConcatSubgraphTest::getTestCaseName);

/* ============= GroupConvolution (DW 3D) ============= */
commonConvParams dwConvParams3D = commonConvParams{kernelSize3D(), strides3D(), padBegin3D(), padEnd3D(), dilation3D(),
                                                   numOutChannels(), paddingType(), numOutChannels()};
const ov::Shape inputShapesDW3D{1, 32, 8, 16, 16};
const std::vector<CPUSpecificParams> CPUParams3D = {
    conv_avx2_dw_3D,
    conv_avx512_dw_3D
};

const auto params3D = ::testing::Combine(
    ::testing::Values(nodeType::groupConvolution),
    ::testing::Values(dwConvParams3D),
    ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams3D)),
    ::testing::Values(inputShapesDW3D),
    ::testing::Values(axis())
);

INSTANTIATE_TEST_SUITE_P(smoke_DWGroupConvolution3D, ConvConcatSubgraphTest, params3D, ConvConcatSubgraphTest::getTestCaseName);

}  // namespace GroupConvolutionDWConcat

namespace ConvolutionBackpropDataConcat {

/* ============= ConvolutionBackpropData (2D) ============= */
const std::vector<CPUSpecificParams> CPUParams2D = {
    block16c_2D
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
    block16c_3D
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

namespace ConvolutionConcat {
/* ============= Convolution (2D) ============= */
const std::vector<CPUSpecificParams> CPUParams2D = {
    conv_avx2_2D,
    conv_avx512_2D
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
    conv_avx2_3D,
    conv_avx512_3D
};

const auto params3D = ::testing::Combine(
    ::testing::Values(nodeType::convolution),
    ::testing::Values(convParams3D()),
    ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams3D)),
    ::testing::Values(inputShapes3D()),
    ::testing::Values(axis())
);

INSTANTIATE_TEST_SUITE_P(smoke_Convolution3D, ConvConcatSubgraphTest, params3D, ConvConcatSubgraphTest::getTestCaseName);
} // namespace ConvolutionConcat

namespace GroupConvolutionConcat {
/* ============= GroupConvolution (2D) ============= */
const std::vector<CPUSpecificParams> CPUParams2D = {
    conv_avx2_2D,
    conv_avx512_2D
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
    conv_avx2_3D,
    conv_avx512_3D
};

const auto params3D = ::testing::Combine(
    ::testing::Values(nodeType::groupConvolution),
    ::testing::Values(groupConvParams3D()),
    ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams3D)),
    ::testing::Values(inputShapes3D()),
    ::testing::Values(axis())
);

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvolution3D, ConvConcatSubgraphTest, params3D, ConvConcatSubgraphTest::getTestCaseName);
} // namespace GroupConvolutionConcat

namespace GroupConvolutionBackpropDataConcat {

/* ============= GroupConvolutionBackpropData (2D) ============= */
const auto params2D = ::testing::Combine(
    ::testing::Values(nodeType::groupConvolutionBackpropData),
    ::testing::Values(groupConvParams2D()),
    ::testing::ValuesIn(filterCPUSpecificParams(blockedCPUParams2D())),
    ::testing::Values(inputShapes2D()),
    ::testing::Values(axis())
);

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvolutionBackpropData2D, ConvConcatSubgraphTest, params2D, ConvConcatSubgraphTest::getTestCaseName);

/* ============= GroupConvolutionBackpropData (3D) ============= */
const auto params3D = ::testing::Combine(
    ::testing::Values(nodeType::groupConvolutionBackpropData),
    ::testing::Values(groupConvParams3D()),
    ::testing::ValuesIn(filterCPUSpecificParams(blockedCPUParams3D())),
    ::testing::Values(inputShapes3D()),
    ::testing::Values(axis())
);

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvolutionBackpropData3D, ConvConcatSubgraphTest, params3D, ConvConcatSubgraphTest::getTestCaseName);

}  // namespace GroupConvolutionBackpropDataConcat
}  // namespace test
}  // namespace ov
