// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_op/softmax.hpp"

namespace ov {
namespace test {
namespace subgraph {

TEST_P(SoftMaxLayerTest, CompareWithRefs) {
    run();
}

TEST_P(SoftMaxLayerTest, CompareQueryModel) {
    query_model();
}

TEST_P(SoftMax8LayerTest, CompareWithRefs) {
    run();
}

TEST_P(SoftMax8LayerTest, CompareQueryModel) {
    query_model();
}

}  // namespace subgraph
}  // namespace test
}  // namespace ov
