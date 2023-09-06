// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_opt/binary_convolution.hpp"

namespace ov {
namespace test {
TEST_P(BinaryConvolutionLayerTest, CompareWithRefs) {
    run();
}
}  // namespace test
}  // namespace ov
