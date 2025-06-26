// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_op/reverse_sequence.hpp"

namespace ov {
namespace test {
TEST_P(ReverseSequenceLayerTest, Inference) {
    run();
};
}  // namespace test
}  // namespace ov
