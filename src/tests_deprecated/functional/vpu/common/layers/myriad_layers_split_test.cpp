// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_split_test.hpp"

INSTANTIATE_TEST_SUITE_P(accuracy, myriadLayersTestsSplit_smoke,
                        ::testing::Values<SplitTestParams>(
                                MAKE_STRUCT(SplitParams, {4, 8, 16, 32, 64}, 2, 6),
                                MAKE_STRUCT(SplitParams, {4, 8, 16, 32}, 2, 6),
                                MAKE_STRUCT(SplitParams, {4, 8, 16}, 1, 6),
                                MAKE_STRUCT(SplitParams, {4, 8}, 0, 3),
                                MAKE_STRUCT(SplitParams, {4}, 0, 3)
                        ));
