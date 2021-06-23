// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_exp_test.hpp"

INSTANTIATE_TEST_SUITE_P(
        accuracy, myriadLayersTestsExp_smoke,
        ::testing::ValuesIn(s_expParams));
