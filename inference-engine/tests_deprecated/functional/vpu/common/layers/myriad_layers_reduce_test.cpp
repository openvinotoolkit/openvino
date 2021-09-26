// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_reduce_test.hpp"

INSTANTIATE_TEST_SUITE_P(accuracy, myriadTestsReduceAnd_smoke,
    ::testing::Combine(
        ::testing::ValuesIn(s_input_pair),
        ::testing::ValuesIn(s_axes_list),
        ::testing::ValuesIn(s_data_precision),
        ::testing::ValuesIn(s_keep_dims))
);

INSTANTIATE_TEST_SUITE_P(accuracy, myriadTestsReduceMin_smoke,
    ::testing::Combine(
        ::testing::ValuesIn(s_input_pair),
        ::testing::ValuesIn(s_axes_list),
        ::testing::ValuesIn(s_data_precision),
        ::testing::ValuesIn(s_keep_dims))
);

INSTANTIATE_TEST_SUITE_P(accuracy, myriadTestsReduceMax_smoke,
    ::testing::Combine(
        ::testing::ValuesIn(s_input_pair),
        ::testing::ValuesIn(s_axes_list),
        ::testing::ValuesIn(s_data_precision),
        ::testing::ValuesIn(s_keep_dims))
);

INSTANTIATE_TEST_SUITE_P(accuracy, myriadTestsReduceSum_smoke,
    ::testing::Combine(
        ::testing::ValuesIn(s_input_pair),
        ::testing::ValuesIn(s_axes_list),
        ::testing::ValuesIn(s_data_precision),
        ::testing::ValuesIn(s_keep_dims))
);

INSTANTIATE_TEST_SUITE_P(accuracy, myriadTestsReduceMean_smoke,
    ::testing::Combine(
        ::testing::ValuesIn(s_input_pair),
        ::testing::ValuesIn(s_axes_list),
        ::testing::ValuesIn(s_data_precision),
        ::testing::ValuesIn(s_keep_dims))
);