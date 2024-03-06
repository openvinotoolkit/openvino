// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_op/ctc_loss.hpp"

namespace ov {
namespace test {
TEST_P(CTCLossLayerTest, Inference) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
}
}  // namespace test
}  // namespace ov
