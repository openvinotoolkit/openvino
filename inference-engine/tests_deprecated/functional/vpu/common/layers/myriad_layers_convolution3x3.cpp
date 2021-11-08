// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_convolution3x3.hpp"

INSTANTIATE_TEST_SUITE_P(myriad, myriadConvolution3x3LayerTests_smoke,
        ::testing::Combine(
        ::testing::Values(CONFIG_VALUE(NO)),
        ::testing::ValuesIn(s_DimsConfig)));
