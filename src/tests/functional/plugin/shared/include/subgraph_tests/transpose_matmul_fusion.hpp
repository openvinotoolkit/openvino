// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/transpose_matmul_fusion.hpp"

namespace ov {
namespace test {

TEST_P(TransposeMatMulFusion, CompareWithRefs){
    run();
};

}  // namespace test
}  // namespace ov
