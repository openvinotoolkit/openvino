// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_strided_slice_test.h"

INSTANTIATE_TEST_SUITE_P(
    accuracy, myriadLayersTestsStridedSlice_smoke,
    ::testing::ValuesIn(s_stridedSliceParams));
