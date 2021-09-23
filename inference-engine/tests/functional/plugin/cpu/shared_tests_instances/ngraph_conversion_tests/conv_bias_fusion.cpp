// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_conversion_tests/conv_bias_fusion.hpp"

#include <vector>

using namespace NGraphConversionTestsDefinitions;

namespace {

INSTANTIATE_TEST_SUITE_P(smoke_Basic, ConvBiasFusion, ::testing::Values("CPU"), ConvBiasFusion::getTestCaseName);

}  // namespace
