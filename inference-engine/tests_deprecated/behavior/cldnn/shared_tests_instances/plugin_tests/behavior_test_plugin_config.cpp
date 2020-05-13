// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior_test_plugin_config.hpp"
#include "cldnn_test_data.hpp"


INSTANTIATE_TEST_CASE_P(smoke_BehaviorTest, BehaviorPluginIncorrectConfigTest, ValuesIn(withIncorrectConfValues),
                        getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_BehaviorTest, BehaviorPluginIncorrectConfigTestInferRequestAPI,
                        ValuesIn(supportedValues),
                        getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_BehaviorTest, BehaviorPluginCorrectConfigTestInferRequestAPI,
                        ValuesIn(supportedValues),
                        getTestCaseName);
