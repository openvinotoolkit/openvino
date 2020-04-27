// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior_test_plugin_exec_graph_info.hpp"
#include "cldnn_test_data.hpp"

// Disabled due to a bug on CentOS that leads to segmentation fault of application on exit
// when perf counters are enabled
//INSTANTIATE_TEST_CASE_P(
//        BehaviorTest,
//        BehaviorPluginTestExecGraphInfo,
//        ValuesIn(supportedValues),
//        getTestCaseName);
