// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior_test_plugin_layers.hpp"

pool_test_params roi_pool_test_cases[] = {
        pool_test_params(CommonTestUtils::DEVICE_CPU, "FP32", pool_case),
};

INSTANTIATE_TEST_CASE_P(smoke_BehaviorTest, ROIPoolingLayerTest,
                        ::testing::ValuesIn(roi_pool_test_cases),
                        getTestName<pool_test_params>);

activ_test_params activ_test_cases[] = {
        activ_test_params(CommonTestUtils::DEVICE_CPU, "FP32", activation_case),
};

INSTANTIATE_TEST_CASE_P(smoke_BehaviorTest, ActivationLayerTest,
                        ::testing::ValuesIn(activ_test_cases),
                        getTestName<activ_test_params>);

INSTANTIATE_TEST_CASE_P(smoke_BehaviorTest, ReLULayerTest,
                        ::testing::Values(activ_test_params("CPU", "FP32", activation_case)),
                        getTestName<activ_test_params>);

norm_test_params norm_test_cases[] = {
        norm_test_params(CommonTestUtils::DEVICE_CPU, "FP32", norm_case),
};

INSTANTIATE_TEST_CASE_P(smoke_BehaviorTest, NormalizeLayerTest,
                        ::testing::ValuesIn(norm_test_cases),
                        getTestName<norm_test_params>);
