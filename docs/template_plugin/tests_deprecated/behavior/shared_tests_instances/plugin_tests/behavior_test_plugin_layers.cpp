// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior_test_plugin_layers.hpp"


conv_test_params deconv_test_cases[] = {
        conv_test_params("TEMPLATE", conv_case),
};

conv_test_params conv_test_cases[] = {
        conv_test_params("TEMPLATE", conv_dw_case),
};


INSTANTIATE_TEST_CASE_P(BehaviorTest, DeconvolutionLayerTest,
                        ::testing::ValuesIn(deconv_test_cases),
                        getTestName<conv_test_params>);

INSTANTIATE_TEST_CASE_P(BehaviorTest, ConvolutionLayerTest,
                        ::testing::ValuesIn(conv_test_cases),
                        getTestName<conv_test_params>);


pool_test_params roi_pool_test_cases[] = {
        pool_test_params("TEMPLATE", "FP32", pool_case),
};

INSTANTIATE_TEST_CASE_P(BehaviorTest, ROIPoolingLayerTest,
                        ::testing::ValuesIn(roi_pool_test_cases),
                        getTestName<pool_test_params>);

activ_test_params activ_test_cases[] = {
        activ_test_params("TEMPLATE", "FP16", activation_case),
};

activ_test_params clamp_test_cases[] = {
    activ_test_params("TEMPLATE", "FP16", clamp_case),
};

INSTANTIATE_TEST_CASE_P(BehaviorTest, ActivationLayerTest,
                        ::testing::ValuesIn(activ_test_cases),
                        getTestName<activ_test_params>);
INSTANTIATE_TEST_CASE_P(BehaviorTest, ReLULayerTest,
                        ::testing::Values(activ_test_params("TEMPLATE", "FP32", activation_case)),
                        getTestName<activ_test_params>);
INSTANTIATE_TEST_CASE_P(BehaviorTest, ClampLayerTest,
    ::testing::ValuesIn(clamp_test_cases),
    getTestName<activ_test_params>);

norm_test_params norm_test_cases[] = {
        norm_test_params("TEMPLATE", "FP32", norm_case),
};

INSTANTIATE_TEST_CASE_P(BehaviorTest, NormalizeLayerTest,
                        ::testing::ValuesIn(norm_test_cases),
                        getTestName<norm_test_params>);

scale_test_params scale_test_cases[] = {
        scale_test_params("TEMPLATE", "FP32", scale_case),
};

INSTANTIATE_TEST_CASE_P(BehaviorTest, ScalingLayerTest,
                        ::testing::ValuesIn(scale_test_cases),
                        getTestName<scale_test_params>);

INSTANTIATE_TEST_CASE_P(BehaviorTest, ShapingLayerTest,
    ::testing::Values(shaping_test_params("TEMPLATE", "FP32", shape_case)),
    getTestName<shaping_test_params>);

INSTANTIATE_TEST_CASE_P(BehaviorTest, ElementWiseLayerTest,
    ::testing::Values(element_test_params("TEMPLATE", "FP32", shape_case)),
    getTestName<element_test_params>);

object_test_params object_test_cases[] = {
        object_test_params("TEMPLATE", "FP32", object_case),
};

INSTANTIATE_TEST_CASE_P(BehaviorTest, ObjectDetectionLayerTest,
                        ::testing::ValuesIn(object_test_cases),
                        getTestName<object_test_params>);

memory_test_params memory_test_cases[] = {
        memory_test_params("TEMPLATE", "FP32", memory_case),
};

// FIXME
// #if (defined INSTANTIATE_TESTS)
// INSTANTIATE_TEST_CASE_P(BehaviorTest, MemoryLayerTest,
//     ::testing::ValuesIn(memory_test_cases),
//     getTestName<memory_test_params>);
// #endif
