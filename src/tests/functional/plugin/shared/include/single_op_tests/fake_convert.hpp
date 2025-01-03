// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_op/fake_convert.hpp"

namespace ov {
namespace test {

TEST_P(FakeConvertLayerTest, Inference) {
    run();
}
}  // namespace test
}  // namespace ov
