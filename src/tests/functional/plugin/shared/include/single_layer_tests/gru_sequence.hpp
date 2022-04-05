// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_layer/gru_sequence.hpp"

namespace LayerTestsDefinitions {

TEST_P(GRUSequenceTest, CompareWithRefs) {
    Run();
};

}  // namespace LayerTestsDefinitions
