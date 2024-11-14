// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/subgraph_tests/src/classes/conv_maxpool_activ.hpp"
#include "common_test_utils/node_builders/convolution.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/fusing_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {
namespace {

const std::vector<fusingSpecificParams> fusingParamsSet{emptyFusingSpec, fusingRelu, fusingSigmoid};

INSTANTIATE_TEST_SUITE_P(smoke_Check,
                         ConvPoolActivTest,
                         ::testing::ValuesIn(fusingParamsSet),
                         ConvPoolActivTest::getTestCaseName);

}  // namespace

}  // namespace test
}  // namespace ov
