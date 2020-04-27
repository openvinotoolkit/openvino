// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_fully_connected_tests.hpp"
INSTANTIATE_TEST_CASE_P(
        accuracy, myriadLayersTestsFullyConnected_nightly,
        ::testing::ValuesIn(s_fcTestParams)
);

INSTANTIATE_TEST_CASE_P(accuracy, myriadLayersTestsFullyConnectedBatch_nightly,
        ::testing::Combine(
            ::testing::ValuesIn(s_fcTestBatchParams)
          , ::testing::ValuesIn(s_fcTestBatchOutSizes)
          )
);

INSTANTIATE_TEST_CASE_P(accuracy, myriadLayersTestsFullyConnectedPVA_nightly,
        ::testing::Combine(
            ::testing::ValuesIn(s_fcTestPVAParams)
          , ::testing::ValuesIn(s_fcTestPVAOutSizes)
          )
);

