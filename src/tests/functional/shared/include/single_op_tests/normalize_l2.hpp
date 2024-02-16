// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "single_op/normalize_l2.hpp"

namespace ov {
namespace test {
TEST_P(NormalizeL2LayerTest, Inference) {
    run();
}
}  // namespace test
}  // namespace ov
