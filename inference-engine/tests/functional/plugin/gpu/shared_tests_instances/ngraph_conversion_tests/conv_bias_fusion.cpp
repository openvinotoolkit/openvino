// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "ngraph_conversion_tests/conv_bias_fusion.hpp"

using namespace NGraphConversionTestsDefinitions;

namespace {

INSTANTIATE_TEST_SUITE_P(smoke_Basic, ConvBiasFusion, ::testing::Values(CommonTestUtils::DEVICE_GPU), ConvBiasFusion::getTestCaseName);

}  // namespace
