// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior_test_plugin_config.hpp"
#include "template_test_data.hpp"

INSTANTIATE_TEST_CASE_P(BehaviorTest, BehaviorPluginIncorrectConfigTest, ValuesIn(withIncorrectConfValues),
                        getTestCaseName);

INSTANTIATE_TEST_CASE_P(BehaviorTest, BehaviorPluginIncorrectConfigTestInferRequestAPI,
                        ValuesIn(supportedValues),
                        getTestCaseName);

INSTANTIATE_TEST_CASE_P(BehaviorTest, BehaviorPluginCorrectConfigTestInferRequestAPI,
                        ValuesIn(supportedValues),
                        getTestCaseName);
