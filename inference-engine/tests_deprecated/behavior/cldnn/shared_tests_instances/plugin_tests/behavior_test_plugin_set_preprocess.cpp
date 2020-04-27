// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior_test_plugin_set_preprocess.hpp"
#include "cldnn_test_data.hpp"

INSTANTIATE_TEST_CASE_P(BehaviorTest,
                        BehaviorPluginTestPreProcess,
                        ValuesIn(supportedValues),
                        getTestCaseName);
