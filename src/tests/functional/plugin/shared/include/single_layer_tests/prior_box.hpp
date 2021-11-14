// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_layer/prior_box.hpp"

namespace LayerTestDefinitions {

TEST_P(PriorBoxLayerTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
}

}  // namespace LayerTestDefinitions
