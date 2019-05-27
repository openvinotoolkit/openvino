// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lstm.hpp"

const char *intel_lstm_projected_layer_name[NUM_LSTM_LAYERS] = {
    "combined input transform",
    "combined recurrent transform",
    "input gate",
    "forget gate",
    "cell gate input part 1",
    "cell gate input part 2",
    "cell gate output part 1",
    "cell gate output part 2",
    "output gate",
    "hidden gated output",
    "projected output"
};

const char *intel_lstm_projected_layer_g4_name[NUM_LSTM_G4_LAYERS] = {
    "combined input transform",
    "deinterleave",
    "interleave 1",
    "interleave 2",
    "interleave 3",
    "interleave 4",
    "combined recurrent transform - 1",
    "input gate - 1",
    "forget gate - 1",
    "cell gate input part 1 - 1",
    "cell gate input part 2 - 1",
    "cell gate output part 1 - 1",
    "cell gate output part 2 - 1",
    "output gate - 1",
    "hidden gated output - 1",
    "projected output - 1",
    "combined recurrent transform - 2",
    "input gate - 2",
    "forget gate - 2",
    "cell gate input part 1 - 2",
    "cell gate input part 2 - 2",
    "cell gate output part 1 - 2",
    "cell gate output part 2 - 2",
    "output gate - 2",
    "hidden gated output - 2",
    "projected output - 2",
    "combined recurrent transform - 3",
    "input gate - 3",
    "forget gate - 3",
    "cell gate input part 1 - 3",
    "cell gate input part 2 - 3",
    "cell gate output part 1 - 3",
    "cell gate output part 2 - 3",
    "output gate - 3",
    "hidden gated output - 3",
    "projected output - 3",
    "combined recurrent transform - 4",
    "input gate - 4",
    "forget gate - 4",
    "cell gate input part 1 - 4",
    "cell gate input part 2 - 4",
    "cell gate output part 1 - 4",
    "cell gate output part 2 - 4",
    "output gate - 4",
    "hidden gated output - 4",
    "projected output - 4",
    "interleave"
};