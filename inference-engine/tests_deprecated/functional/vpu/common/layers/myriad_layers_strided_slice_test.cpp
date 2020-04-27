// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_strided_slice_test.h"

INSTANTIATE_TEST_CASE_P(
    accuracy, myriadLayersTestsStridedSlice_nightly,
    ::testing::ValuesIn(s_stridedSliceParams));
