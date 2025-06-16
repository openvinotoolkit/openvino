// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_op/multiclass_nms.hpp"

namespace ov {
namespace test {
TEST_P(MulticlassNmsLayerTest, Inference) {
    run();
};

TEST_P(MulticlassNmsLayerTest8, Inference) {
    run();
};
} // namespace test
} // namespace ov
