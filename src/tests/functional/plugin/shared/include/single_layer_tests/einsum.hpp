// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_layer/einsum.hpp"

namespace LayerTestsDefinitions {

TEST_P(EinsumLayerTest, CompareWithRefs) {
    Run();
}

}  // namespace LayerTestsDefinitions
