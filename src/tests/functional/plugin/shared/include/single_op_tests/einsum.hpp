// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_op/einsum.hpp"

namespace ov {
namespace test {
TEST_P(EinsumLayerTest, Inference) {
    run();
}
}  // namespace test
}  // namespace ov
