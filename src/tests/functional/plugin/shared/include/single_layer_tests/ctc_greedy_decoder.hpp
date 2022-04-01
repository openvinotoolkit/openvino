// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_layer/ctc_greedy_decoder.hpp"

namespace LayerTestsDefinitions {

TEST_P(CTCGreedyDecoderLayerTest, CompareWithRefs) {
    Run();
};

TEST_P(CTCGreedyDecoderLayerTest, QueryNetwork) {
    QueryNetwork();
}
}  // namespace LayerTestsDefinitions
