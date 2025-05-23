// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/stridedslice_eltwise.hpp"

namespace ov {
namespace test {
TEST_P(StridedSliceEltwiseTest, Inference) {
    run();
};
}  // namespace test
}  // namespace ov
