// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_select_test.hpp"

INSTANTIATE_TEST_SUITE_P(accuracy, myriadTestsSelect_smoke,
    ::testing::Combine(
        ::testing::ValuesIn(s_eltwiseTensors),
        ::testing::ValuesIn(s_eltwiseDims))
);
