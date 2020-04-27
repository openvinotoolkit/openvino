// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_lrn_test.hpp"

INSTANTIATE_TEST_CASE_P(accuracy, myriadLayersTestsLRN_nightly,
    ::testing::Combine(
        ::testing::ValuesIn(s_LRNTensors),
        ::testing::ValuesIn(s_LRNlocal_size),
        ::testing::ValuesIn(s_LRN_K),
        ::testing::ValuesIn(s_LRNalpha),
        ::testing::ValuesIn(s_LRNbeta))
);
