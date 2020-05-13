// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior_test_plugin_layers.hpp"
#include "gna_test_data.hpp"


conv_test_params deconv_test_cases[] = {
        conv_test_params(CommonTestUtils::DEVICE_GNA, conv_case)
};

INSTANTIATE_TEST_CASE_P(smoke_BehaviorTest, DeconvolutionLayerTest,
                        ::testing::ValuesIn(deconv_test_cases),
                        getTestName<conv_test_params>);

pool_test_params roi_pool_test_cases[] = {
        pool_test_params(CommonTestUtils::DEVICE_GNA, "FP32", pool_case),
};

// TODO: fix this
//INSTANTIATE_TEST_CASE_P(smoke_BehaviorTest, PoolingLayerTest,
//                        ::testing::Values(pool_test_params("GNAPlugin", "FP32", pool_case)),
//                        getTestName<pool_test_params>);
//
//INSTANTIATE_TEST_CASE_P(smoke_BehaviorTest, ReLULayerTest,
//                        ::testing::Values(activ_test_params("GNAPlugin", "FP32", activation_case)),
//                        getTestName<activ_test_params>);

// FIXME
//#if (defined INSTANTIATE_TESTS)
//INSTANTIATE_TEST_CASE_P(smoke_BehaviorTest, MemoryLayerTest,
//    ::testing::ValuesIn(memory_test_cases),
//    getTestName<memory_test_params>);
//#endif
