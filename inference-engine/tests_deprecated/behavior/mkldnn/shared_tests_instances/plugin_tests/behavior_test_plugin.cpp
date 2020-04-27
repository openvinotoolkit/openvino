// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior_test_plugin.h"
#include "behavior_test_plugins.hpp"
#include "mkldnn_test_data.hpp"

INSTANTIATE_TEST_CASE_P(BehaviorTest, BehaviorPluginTest, ValuesIn(supportedValues),
                        getTestCaseName);
INSTANTIATE_TEST_CASE_P(BehaviorTest, BehaviorPluginTestInput, ValuesIn(allInputSupportedValues),
                        getTestCaseName);
INSTANTIATE_TEST_CASE_P(BehaviorTest, BehaviorPluginTestOutput, ValuesIn(allOutputSupportedValues),
                        getOutputTestCaseName);
