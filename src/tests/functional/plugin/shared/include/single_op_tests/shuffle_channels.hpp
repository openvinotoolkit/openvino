// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_op/shuffle_channels.hpp"

namespace ov {
namespace test {
TEST_P(ShuffleChannelsLayerTest, Inference) {
    run();
}
}  // namespace test
}  // namespace ov
