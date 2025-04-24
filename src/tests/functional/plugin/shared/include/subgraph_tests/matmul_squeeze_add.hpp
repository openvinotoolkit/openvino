// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/matmul_squeeze_add.hpp"

namespace ov {
namespace test {

TEST_P(MatmulSqueezeAddTest, CompareWithRefImpl) {
    run();
};

}  // namespace test
}  // namespace ov
