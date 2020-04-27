// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_exp_test.hpp"

INSTANTIATE_TEST_CASE_P(
        accuracy, myriadLayersTestsExp_nightly,
        ::testing::ValuesIn(s_expParams));
