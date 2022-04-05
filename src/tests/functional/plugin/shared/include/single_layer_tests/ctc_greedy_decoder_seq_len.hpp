// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_layer/ctc_greedy_decoder_seq_len.hpp"

namespace LayerTestsDefinitions {

TEST_P(CTCGreedyDecoderSeqLenLayerTest, CompareWithRefs) {
    Run();
};

}  // namespace LayerTestsDefinitions
