// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior_test_plugin_layers.hpp"

pool_test_params roi_pool_test_cases[] = {
    pool_test_params(CommonTestUtils::DEVICE_MYRIAD, "FP16", pool_case),
};

INSTANTIATE_TEST_CASE_P(BehaviorTest, ROIPoolingLayerTest,
                        ::testing::ValuesIn(roi_pool_test_cases),
                        getTestName<pool_test_params>);

memory_test_params memory_test_cases[] = {
    memory_test_params(CommonTestUtils::DEVICE_MYRIAD, "FP32", memory_case),
};

// FIXME
//#if (defined INSTANTIATE_TESTS)
//INSTANTIATE_TEST_CASE_P(BehaviorTest, MemoryLayerTest,
//    ::testing::ValuesIn(memory_test_cases),
//    getTestName<memory_test_params>);
//#endif

