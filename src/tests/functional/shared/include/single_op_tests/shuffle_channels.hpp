// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "single_op/shuffle_channels.hpp"

namespace ov {
namespace test {
TEST_P(ShuffleChannelsLayerTest, Inference) {
    run();
}
}  // namespace test
}  // namespace ov
