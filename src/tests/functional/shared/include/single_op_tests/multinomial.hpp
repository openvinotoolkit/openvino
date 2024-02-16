// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "single_op/multinomial.hpp"

namespace ov {
namespace test {
TEST_P(MultinomialLayerTest, Inference) {
    run();
};
}  // namespace test
}  // namespace ov
