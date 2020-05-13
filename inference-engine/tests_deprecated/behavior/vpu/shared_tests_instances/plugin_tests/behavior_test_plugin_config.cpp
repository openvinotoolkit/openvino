// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior_test_plugin_config.hpp"
#include "vpu_test_data.hpp"

INSTANTIATE_TEST_CASE_P(
    smoke_BehaviorTest, BehaviorPluginCorrectConfigTest,
    ValuesIn(
        BehTestParams::concat(
            BehTestParams::concat(deviceSpecificConfigurations, deviceAgnosticConfigurations),
            withCorrectConfValuesPluginOnly
        )
    ),
    getTestCaseName
);

INSTANTIATE_TEST_CASE_P(smoke_BehaviorTest, BehaviorPluginIncorrectConfigTest, ValuesIn(withIncorrectConfValues),
                        getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_BehaviorTest, BehaviorPluginIncorrectConfigTestInferRequestAPI,
                        ValuesIn(withIncorrectConfKeys),
                        getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_BehaviorTest, BehaviorPluginCorrectConfigTestInferRequestAPI,
                        ValuesIn(supportedValues),
                        getTestCaseName);
