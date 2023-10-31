// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_op/nms_rotated.hpp"

namespace LayerTestsDefinitions {

TEST_P(NmsRotatedOpTest, CompareWithRefs) {
    run();
};

}  // namespace LayerTestsDefinitions
