// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/quantized_group_convolution.hpp"

namespace ov {
namespace test {

TEST_P(QuantGroupConvLayerTest, CompareWithRefs) {
    run();
}

}  // namespace test
}  // namespace ov
