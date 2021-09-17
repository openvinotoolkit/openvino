// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_power_test.hpp"
INSTANTIATE_TEST_SUITE_P( accuracy, myriadLayersTestsPowerParams_smoke,
    ::testing::Combine(
        ::testing::ValuesIn(s_powerTensors),
        ::testing::ValuesIn(s_powerParams))
);
