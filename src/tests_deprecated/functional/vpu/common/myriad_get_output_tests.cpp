// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_get_output_tests.hpp"

INSTANTIATE_TEST_SUITE_P(Test_params_pool, myriadGetOutput_nightly,
        testing::Values(
                std::make_tuple(std::make_tuple(&full_model, &poolModel), "pool1_3x3_s2"),
                std::make_tuple(std::make_tuple(&full_model, &convModel), "conv1_7x7_s2"),
                std::make_tuple(std::make_tuple(&full_model, &reluConvModel), "conv1_relu_7x7"),
                std::make_tuple(std::make_tuple(&full_model, &fcModel), "loss3_classifier"),
                std::make_tuple(std::make_tuple(&full_model, &reluFcModel), "ReluFC"),
                std::make_tuple(std::make_tuple(&concatModel, &concatModelConv), "conv1_2")
        ),
    getTestCaseName
);
