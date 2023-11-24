// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/src/classes/conv_concat.hpp"
#include "test_utils/convolution_params.hpp"
#include "test_utils/fusing_test_utils.hpp"
#include "test_utils/filter_cpu_info.hpp"
#include "ov_models/builders.hpp"

using namespace ngraph;
using namespace InferenceEngine;
using namespace CPUTestUtils;

namespace SubgraphTestsDefinitions {
namespace ConvConcat {

/* ============= Convolution (3D) ============= */
const std::vector<CPUSpecificParams> CPUParams3D = {
    conv_ref_3D
};

const auto params3D = ::testing::Combine(
    ::testing::Values(nodeType::convolution),
    ::testing::Values(convParams3D()),
    ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams3D)),
    ::testing::Values(inputShapes3D()),
    ::testing::Values(axis())
);

INSTANTIATE_TEST_SUITE_P(smoke_Convolution3D, ConvConcatSubgraphTest, params3D, ConvConcatSubgraphTest::getTestCaseName);

const auto params3D = ::testing::Combine(
    ::testing::Values(nodeType::groupConvolution),
    ::testing::Values(groupConvParams3D()),
    ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams3D)),
    ::testing::Values(inputShapes3D()),
    ::testing::Values(axis())
);

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvolution3D, ConvConcatSubgraphTest, params3D, ConvConcatSubgraphTest::getTestCaseName);
} // namespace ConvConcat
} // namespace SubgraphTestsDefinitions