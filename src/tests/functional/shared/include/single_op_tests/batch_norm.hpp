// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "single_op/batch_norm.hpp"

namespace ov {
namespace test {
TEST_P(BatchNormLayerTest, Inference) {
    run();
}
}  // namespace test
}  // namespace ov
