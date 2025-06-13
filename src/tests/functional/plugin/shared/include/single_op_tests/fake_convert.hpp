// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "single_op/fake_convert.hpp"

namespace ov {
namespace test {

TEST_P(FakeConvertLayerTest, Inference) {
    run();
}
}  // namespace test
}  // namespace ov
