// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_layer/gru_cell.hpp"

namespace LayerTestsDefinitions {

TEST_P(GRUCellTest, CompareWithRefs) {
    Run();
};

}  // namespace LayerTestsDefinitions
