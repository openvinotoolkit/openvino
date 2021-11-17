// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_layer/bucketize.hpp"

namespace ov {
namespace test {
namespace subgraph {

TEST_P(BucketizeLayerTest, CompareWithRefs) {
    run();
}

}  // namespace subgraph
}  // namespace test
}  // namespace ov
