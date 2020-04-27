// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_reduce_test.hpp"

INSTANTIATE_TEST_CASE_P(accuracy, myriadTestsReduceAnd_nightly,
    ::testing::Combine(
        ::testing::ValuesIn(s_input_dims),
        ::testing::ValuesIn(s_axes_list),
        ::testing::ValuesIn(s_data_precision),
        ::testing::ValuesIn(s_keep_dims))
);

INSTANTIATE_TEST_CASE_P(accuracy, myriadTestsReduceMin_nightly,
    ::testing::Combine(
        ::testing::ValuesIn(s_input_dims),
        ::testing::ValuesIn(s_axes_list),
        ::testing::ValuesIn(s_data_precision),
        ::testing::ValuesIn(s_keep_dims))
);

INSTANTIATE_TEST_CASE_P(accuracy, myriadTestsReduceMax_nightly,
    ::testing::Combine(
        ::testing::ValuesIn(s_input_dims),
        ::testing::ValuesIn(s_axes_list),
        ::testing::ValuesIn(s_data_precision),
        ::testing::ValuesIn(s_keep_dims))
);

INSTANTIATE_TEST_CASE_P(accuracy, myriadTestsReduceSum_nightly,
    ::testing::Combine(
        ::testing::ValuesIn(s_input_dims),
        ::testing::ValuesIn(s_axes_list),
        ::testing::ValuesIn(s_data_precision),
        ::testing::ValuesIn(s_keep_dims))
);

INSTANTIATE_TEST_CASE_P(accuracy, myriadTestsReduceMean_nightly,
    ::testing::Combine(
        ::testing::ValuesIn(s_input_dims),
        ::testing::ValuesIn(s_axes_list),
        ::testing::ValuesIn(s_data_precision),
        ::testing::ValuesIn(s_keep_dims))
);
