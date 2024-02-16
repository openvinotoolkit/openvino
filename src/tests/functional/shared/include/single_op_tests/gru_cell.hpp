// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "single_op/gru_cell.hpp"

namespace ov {
namespace test {
TEST_P(GRUCellTest, Inference) {
    run();
};
}  // namespace test
}  // namespace ov
