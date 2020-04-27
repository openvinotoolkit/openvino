// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_resample_test.hpp"

INSTANTIATE_TEST_CASE_P(myriad, myriadResampleLayerTests_nightly,
        ::testing::Combine(
        ::testing::Values(CONFIG_VALUE(NO), CONFIG_VALUE(YES)),
        ::testing::ValuesIn(s_ResampleCustomConfig),
        ::testing::ValuesIn(s_ResampleAntialias)));
