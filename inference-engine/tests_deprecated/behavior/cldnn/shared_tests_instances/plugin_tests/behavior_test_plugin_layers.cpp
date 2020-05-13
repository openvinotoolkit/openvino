// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior_test_plugin_layers.hpp"

memory_test_params memory_test_cases[] = {
        memory_test_params("GPU", "FP32", memory_case),
};

// FIXME
//#if (defined INSTANTIATE_TESTS)
//INSTANTIATE_TEST_CASE_P(smoke_BehaviorTest, MemoryLayerTest,
//    ::testing::ValuesIn(memory_test_cases),
//    getTestName<memory_test_params>);
//#endif
