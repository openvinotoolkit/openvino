// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_opt/bucketize.hpp"

namespace ov {
namespace test {
TEST_P(BucketizeLayerTest, CompareWithRefs) {
    run();
}
}  // namespace test
}  // namespace ov
