// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "single_op/transpose.hpp"

namespace ov {
namespace test {
TEST_P(TransposeLayerTest, Inference) {
    run();
};
}  // namespace test
}  // namespace ov