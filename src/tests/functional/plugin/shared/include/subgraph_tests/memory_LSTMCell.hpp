// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/memory_LSTMCell.hpp"

namespace ov {
namespace test {

TEST_P(MemoryLSTMCellTest, CompareWithRefs) {
    run();
};

}  // namespace test
}  // namespace ov
