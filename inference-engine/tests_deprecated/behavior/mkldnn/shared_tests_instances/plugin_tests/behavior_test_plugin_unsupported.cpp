// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior_test_plugin_unsupported.hpp"
#include "mkldnn_test_data.hpp"

INSTANTIATE_TEST_CASE_P(smoke_BehaviorTest, BehaviorPluginTestAllUnsupported, ValuesIn(allUnSupportedValues),
    getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_BehaviorTest, BehaviorPluginTestTypeUnsupported, ValuesIn(typeUnSupportedValues),
    getTestCaseName);
