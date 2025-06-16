// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/relu_shape_of.hpp"

namespace ov {
namespace test {

TEST_P(ReluShapeOfSubgraphTest, CompareWithRefs) {
    run();
}

}  // namespace test
}  // namespace ov
