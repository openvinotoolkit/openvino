// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior_test_plugin_exec_graph_info.hpp"
#include "vpu_test_data.hpp"

// TODO: currently this tests are not applicable to myriadPlugin
#if 0
INSTANTIATE_TEST_CASE_P(smoke_
        BehaviorTest,
        BehaviorPluginTestExecGraphInfo,
        ValuesIn(supportedValues),
        getTestCaseName);
#endif