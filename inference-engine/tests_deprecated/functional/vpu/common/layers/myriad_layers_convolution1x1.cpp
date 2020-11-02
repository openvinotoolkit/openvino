// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_convolution1x1.hpp"

INSTANTIATE_TEST_CASE_P(myriad, myriadConvolution1x1LayerTests_smoke,
        ::testing::Combine(
        ::testing::Values(CONFIG_VALUE(NO)),
        ::testing::ValuesIn(s_isHWC),
        ::testing::ValuesIn(s_DimsConfig)));
