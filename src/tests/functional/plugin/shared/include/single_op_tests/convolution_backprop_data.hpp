// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// DEPRECATED, can't be removed currently due to arm and kmb-plugin dependency (#55568)
#pragma once

#include "shared_test_classes/single_op/convolution_backprop_data.hpp"

namespace ov {
namespace test {
TEST_P(ConvolutionBackpropDataLayerTest, Inference) {
    run();
}
} // namespace test
} // namespace ov
