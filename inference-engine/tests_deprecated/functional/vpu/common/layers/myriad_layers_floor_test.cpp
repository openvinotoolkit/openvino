// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_floor_test.hpp"

INSTANTIATE_TEST_SUITE_P(
        accuracy, myriadLayersTestsFloor_smoke,
        ::testing::ValuesIn(s_FloorParams));
