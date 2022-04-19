// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_layer/lstm_cell_basic.hpp"

namespace LayerTestsDefinitions {

TEST_P(LSTMCellBasicTest, CompareWithRefs) {
    Run();
};

}  // namespace LayerTestsDefinitions
