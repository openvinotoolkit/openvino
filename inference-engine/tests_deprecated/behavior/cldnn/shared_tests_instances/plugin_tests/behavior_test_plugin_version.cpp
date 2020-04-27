// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior_test_plugin_version.hpp"
#include "cldnn_test_data.hpp"

INSTANTIATE_TEST_CASE_P(BehaviorTest, BehaviorPluginTestVersion, ValuesIn(add_element_into_array(supportedValues, BEH_HETERO)), getTestCaseName);
