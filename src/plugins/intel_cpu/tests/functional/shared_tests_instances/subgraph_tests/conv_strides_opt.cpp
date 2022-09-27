// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "subgraph_tests/conv_strides_opt.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace SubgraphTestsDefinitions;

namespace {
    std::vector<ngraph::Shape> input_shapes{
        ngraph::Shape{1, 1, 4, 4},
        ngraph::Shape{1, 64, 56, 56},
    };
    std::vector<ngraph::op::PadType> pads{
        ngraph::op::PadType::SAME_UPPER,
        ngraph::op::PadType::SAME_LOWER,
        ngraph::op::PadType::EXPLICIT,
    };
    INSTANTIATE_TEST_SUITE_P(smoke_Convolution_StridesOpt, ConvStridesOpt,
                            ::testing::Combine(
                                    ::testing::ValuesIn(input_shapes),
                                    ::testing::ValuesIn(pads),
                                    ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                            ConvStridesOpt::getTestCaseName);

}  // namespace
