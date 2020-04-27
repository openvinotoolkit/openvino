// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_select_test.hpp"

INSTANTIATE_TEST_CASE_P(accuracy, myriadTestsSelect_nightly,
    ::testing::Combine(
        ::testing::ValuesIn(s_eltwiseTensors),
        ::testing::ValuesIn(s_eltwiseDims))
);
