// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_layer/prior_box.hpp"

namespace LayerTestDefinitions {

TEST_P(PriorBoxLayerTest, CompareWithRefs) {
    Run();
}

}  // namespace LayerTestDefinitions
