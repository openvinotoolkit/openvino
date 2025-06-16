// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/reshape_permute_conv_permute_reshape_act.hpp"

namespace ov {
namespace test {
TEST_P(ConvReshapeAct, Inference) {
    run();
}
}  // namespace test
}  // namespace ov
