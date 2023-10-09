// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/constant_result.hpp"

namespace ov {
namespace test {

TEST_P(ConstantResultSubgraphTest, CompareWithRefs) {
    run();
}

}  // namespace test
}  // namespace ov
