// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "subgraph/multiple_LSTMCell.hpp"

namespace ov {
namespace test {

TEST_P(MultipleLSTMCellTest, CompareWithRefs) {
    run();
};

}  // namespace test
}  // namespace ov
