// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior_test_plugin_unsupported.hpp"
#include "template_test_data.hpp"

// INSTANTIATE_TEST_CASE_P(BehaviorTest, BehaviorPluginTestAllUnsupported, ValuesIn(allUnSupportedValues),
//     getTestCaseName);

INSTANTIATE_TEST_CASE_P(BehaviorTest, BehaviorPluginTestTypeUnsupported, ValuesIn(typeUnSupportedValues),
    getTestCaseName);
INSTANTIATE_TEST_CASE_P(BehaviorTest, BehaviorPluginTestBatchUnsupported, ValuesIn(batchUnSupportedValues),
    getTestCaseName);
