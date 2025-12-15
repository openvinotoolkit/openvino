// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/matmul_compressed_weights.hpp"

namespace ov {
namespace test {

TEST_P(MatmulCompressedTest, CompareWithRefImpl) {
    run();
};

}  // namespace test
}  // namespace ov
