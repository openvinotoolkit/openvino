// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_log_test.hpp"

INSTANTIATE_TEST_CASE_P(
        accuracy, myriadLayersTestsLog_nightly,
        ::testing::ValuesIn(s_logParams));
