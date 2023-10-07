// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_layer/batch_norm.hpp"
#include "ov_models/builders.hpp"

namespace LayerTestsDefinitions {

TEST_P(BatchNormLayerTest, CompareWithRefs) {
    Run();
}

}  // namespace LayerTestsDefinitions
