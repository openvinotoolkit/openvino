// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "single_op/istft.hpp"

namespace ov {
namespace test {
TEST_P(ISTFTLayerTest, CompareWithRefs) {
    run();
};
}  // namespace test
}  // namespace ov
