// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "single_op/eye.hpp"

namespace ov {
namespace test {
TEST_P(EyeLayerTest, Inference) {
    run();
}
}  // namespace test
}  // namespace ov
