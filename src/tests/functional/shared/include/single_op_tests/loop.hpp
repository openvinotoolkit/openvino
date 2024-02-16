// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "single_op/loop.hpp"

namespace ov {
namespace test {
TEST_P(LoopLayerTest, Inference) {
    run();
}

TEST_P(StaticShapeLoopLayerTest, Inference) {
    run();
}
}  // namespace test
}  // namespace ov
