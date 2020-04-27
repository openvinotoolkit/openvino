// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_gather_test.hpp"

using namespace testing;

INSTANTIATE_TEST_CASE_P(accuracy, myriadLayerGather_nightly,
                        Values(GatherTestParams { {36549, 1024},   {16},           0, "FP16" },
                               GatherTestParams { {10},            {10},           0, "FP16" },
                               GatherTestParams { {36549, 1024},   {10},           0, "FP16" },
                               GatherTestParams { {365490},        {10},           0, "FP16" },
                               GatherTestParams { {10, 1024},      {10},           0, "FP16" },
                               GatherTestParams { {30522, 768},    {1, 128, 1},    0, "FP16" },
                               GatherTestParams { {30522, 768},    {1, 128, 1},    1, "FP16" },
                               GatherTestParams { {6, 12, 10, 24}, {15, 4, 20, 5}, 0, "FP16" },
                               GatherTestParams { {6, 12, 10, 24}, {15, 4, 20, 5}, 1, "FP16" },
                               GatherTestParams { {6, 12, 10, 24}, {15, 4, 20, 5}, 2, "FP16" },
                               GatherTestParams { {6, 12, 10, 24}, {15, 4, 20, 5}, 3, "FP16" },
                               GatherTestParams { {10},            {10},           0, "I32" },
                               GatherTestParams { {365490},        {10},           0, "I32" },
                               GatherTestParams { {36549, 768},    {10},           0, "I32" },
                               GatherTestParams { {30522, 768},    {1, 128, 1},    0, "I32" },
                               GatherTestParams { {30522, 768},    {1, 128, 1},    1, "I32" },
                               GatherTestParams { {6, 12, 10, 24}, {15, 4, 20, 5}, 0, "I32" },
                               GatherTestParams { {6, 12, 10, 24}, {15, 4, 20, 5}, 3, "I32" }));
