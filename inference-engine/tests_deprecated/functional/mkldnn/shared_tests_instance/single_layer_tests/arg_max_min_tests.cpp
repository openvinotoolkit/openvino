// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "arg_max_min_tests.hpp"

static std::vector<float> in_data = { 0.0f, 1.0f,
                                  20.0f, 12.0f,

                                  12.0f, 0.0f,
                                  15.0f, 8.0f,

                                  9.0f, 4.0f,
                                  25.0f, 15.0f,


                                  0.0f, 0.0f,
                                  1.0f, 1.0f,

                                  0.0f, 0.0f,
                                  24.0f, 12.0f,

                                  8.0f, 9.0f,
                                  2.0f, 14.0 };

INSTANTIATE_TEST_CASE_P(
        smoke_mkldnn_TestsArgMaxMin, ArgMaxMinTFTests,
        ::testing::Values(
                // Params: device_name, in_dim, in_data, has_axis, out_max_val, top_k, axis, ref_dim, ref_data
                argMaxMinTF_test_params{ "CPU", "ArgMax", { 2, 3, 2, 2 }, in_data,
                                                                     1, 0, 1, 0, { 1, 3, 2, 2 } },

                argMaxMinTF_test_params{ "CPU", "ArgMax", { 2, 3, 2, 2 }, in_data,
                                                                     1, 0, 1, 1, { 2, 1, 2, 2 } },

                argMaxMinTF_test_params{ "CPU", "ArgMax", { 2, 3, 2, 2 }, in_data,
                                                                     1, 0, 1, 2, { 2, 3, 1, 2 } },

                argMaxMinTF_test_params{ "CPU", "ArgMax", { 2, 3, 2, 2 }, in_data,
                                                                     1, 0, 1, 3, { 2, 3, 2, 1 } },

                argMaxMinTF_test_params{ "CPU", "ArgMax", { 2, 3, 2, 2 }, in_data,
                                                                     1, 0, 2, 0, { 2, 3, 2, 2 } },

                argMaxMinTF_test_params{ "CPU", "ArgMax", { 2, 3, 2, 2 }, in_data,
                                                                     1, 0, 2, 1, { 2, 2, 2, 2 } },

                argMaxMinTF_test_params{ "CPU", "ArgMax", { 2, 3, 2, 2 }, in_data,
                                                                     1, 0, 2, 2, { 2, 3, 2, 2 } },

                argMaxMinTF_test_params{ "CPU", "ArgMax", { 2, 3, 2, 2 }, in_data,
                                                                     1, 0, 2, 3, { 2, 3, 2, 2 } },

                argMaxMinTF_test_params{ "CPU", "ArgMax", { 2, 3, 2, 2 }, in_data,
                                                                     1, 0, 3, 1, { 2, 3, 2, 2 } }
        ));
