// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_slice_test.hpp"

INSTANTIATE_TEST_SUITE_P(accuracy, myriadLayersTestsSlice_smoke,
                        ::testing::Values<SliceTestParams>(
                                MAKE_STRUCT(SliceParams, {4, 8, 16, 32, 64}, {{4, 8, 16, 10, 64}, {4, 8, 16, 22, 64}}, 3),
                                MAKE_STRUCT(SliceParams, {4, 8, 16, 32}, {{4, 8, 2, 32}, {4, 8, 14, 32}}, 2))
);
