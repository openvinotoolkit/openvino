// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_layer/normalize_l2.hpp"

namespace LayerTestsDefinitions {

TEST_P(NormalizeL2LayerTest, CompareWithRefs) {
    Run();
}

}  // namespace LayerTestsDefinitions
