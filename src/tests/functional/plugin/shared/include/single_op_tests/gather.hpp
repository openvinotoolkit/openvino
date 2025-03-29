// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_op/gather.hpp"

namespace ov {
namespace test {
TEST_P(GatherLayerTest, Inference) {
    run();
};

TEST_P(Gather7LayerTest, Inference) {
    run();
};

TEST_P(Gather8LayerTest, Inference) {
    run();
};

TEST_P(Gather8IndiceScalarLayerTest, Inference) {
    run();
};

TEST_P(Gather8withIndicesDataLayerTest, Inference) {
    run();
};

TEST_P(GatherStringWithIndicesDataLayerTest, Inference) {
    run();
};

}  // namespace test
}  // namespace ov
