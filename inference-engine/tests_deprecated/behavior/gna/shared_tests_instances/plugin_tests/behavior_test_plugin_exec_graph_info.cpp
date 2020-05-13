// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior_test_plugin_exec_graph_info.hpp"
#include "gna_test_data.hpp"

INSTANTIATE_TEST_CASE_P(
        smoke_BehaviorTest,
        BehaviorPluginTestExecGraphInfo,
        ValuesIn(supportedValues),
        getTestCaseName);
