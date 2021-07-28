// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_pooling_test.hpp"

INSTANTIATE_TEST_SUITE_P(accuracy, myriadLayers_IR3_BatchPoolingTests_smoke,
                        ::testing::Combine(
                                ::testing::Values<InferenceEngine::SizeVector>({10, 192, 56, 56})
                                , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 3, 3)) /* kernel     */
                                , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 2, 2)) /* stride     */
                                , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 0, 0)) /* pads_begin */
                                , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 1, 1)) /* pads_end   */
                                , ::testing::ValuesIn(s_poolingAutoPad)
                                , ::testing::ValuesIn(s_poolingExcludePad)
                                , ::testing::ValuesIn(s_poolingMethod)
                        )
);

INSTANTIATE_TEST_SUITE_P(accuracy_1, myriadLayers_IR3_BatchPoolingTests_smoke,
                        ::testing::Combine(
                                ::testing::Values<InferenceEngine::SizeVector>({10, 576, 14, 14})
                                , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 2, 2)) /* kernel     */
                                , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 2, 2)) /* stride     */
                                , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 0, 0)) /* pads_begin */
                                , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 0, 0)) /* pads_end   */
                                , ::testing::ValuesIn(s_poolingAutoPad)
                                , ::testing::ValuesIn(s_poolingExcludePad)
                                , ::testing::ValuesIn(s_poolingMethod)
                        )
);


INSTANTIATE_TEST_SUITE_P(accuracy_4X4, myriadLayers_IR3_PoolingTests_smoke,
                        ::testing::Combine(
                                ::testing::Values<InferenceEngine::SizeVector>({10, 1024, 4, 4})
                                , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 4, 4)) /* kernel     */
                                , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 1, 1)) /* stride     */
                                , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 0, 0)) /* pads_begin */
                                , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 0, 0)) /* pads_end   */
                                , ::testing::ValuesIn(s_poolingAutoPad)
                                , ::testing::ValuesIn(s_poolingExcludePad)
                                , ::testing::ValuesIn(s_poolingMethod)
                        )
);

INSTANTIATE_TEST_SUITE_P(accuracy_1X1, myriadLayers_IR3_BatchPoolingTests_smoke,
                        ::testing::Combine(
                                ::testing::Values<InferenceEngine::SizeVector>({1, 3, 5, 7})
                                , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 1, 1)) /* kernel     */
                                , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 1, 1)) /* stride     */
                                , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 0, 0)) /* pads_begin */
                                , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 0, 0)) /* pads_end   */
                                , ::testing::ValuesIn(s_poolingAutoPad)
                                , ::testing::ValuesIn(s_poolingExcludePad)
                                , ::testing::ValuesIn(s_poolingMethod)
                        )
);

INSTANTIATE_TEST_SUITE_P(accuracy_2X2p0000, myriadLayers_IR3_BatchPoolingTests_smoke,
                        ::testing::Combine(
                                ::testing::Values<InferenceEngine::SizeVector>({1, 512, 26, 26})
                                , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 2, 2)) /* kernel     */
                                , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 1, 1)) /* stride     */
                                , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 0, 0)) /* pads_begin */
                                , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 0, 0)) /* pads_end   */
                                , ::testing::ValuesIn(s_poolingAutoPad)
                                , ::testing::ValuesIn(s_poolingExcludePad)
                                , ::testing::ValuesIn(s_poolingMethod)
                        )
);

INSTANTIATE_TEST_SUITE_P(accuracy_2X2p0001, myriadLayers_IR3_BatchPoolingTests_smoke,
                        ::testing::Combine(
                                ::testing::Values<InferenceEngine::SizeVector>({1, 512, 26, 26})
                                , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 2, 2)) /* kernel     */
                                , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 1, 1)) /* stride     */
                                , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 0, 0)) /* pads_begin */
                                , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 0, 1)) /* pads_end   */
                                , ::testing::ValuesIn(s_poolingAutoPad)
                                , ::testing::ValuesIn(s_poolingExcludePad)
                                , ::testing::ValuesIn(s_poolingMethod)
                        )
);
INSTANTIATE_TEST_SUITE_P(accuracy_2X2p0011, myriadLayers_IR3_BatchPoolingTests_smoke,
                        ::testing::Combine(
                                ::testing::Values<InferenceEngine::SizeVector>({1, 512, 26, 26})
                                , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 2, 2)) /* kernel     */
                                , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 1, 1)) /* stride     */
                                , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 0, 0)) /* pads_begin */
                                , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 1, 1)) /* pads_end   */
                                , ::testing::ValuesIn(s_poolingAutoPad)
                                , ::testing::ValuesIn(s_poolingExcludePad)
                                , ::testing::ValuesIn(s_poolingMethod)
                        )
);
INSTANTIATE_TEST_SUITE_P(accuracy_2X2p0111, myriadLayers_IR3_BatchPoolingTests_smoke,
                        ::testing::Combine(
                                ::testing::Values<InferenceEngine::SizeVector>({1, 512, 26, 26})
                                , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 2, 2)) /* kernel     */
                                , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 1, 1)) /* stride     */
                                , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 0, 1)) /* pads_begin */
                                , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 1, 1)) /* pads_end   */
                                , ::testing::ValuesIn(s_poolingAutoPad)
                                , ::testing::ValuesIn(s_poolingExcludePad)
                                , ::testing::ValuesIn(s_poolingMethod)
                        )
);
INSTANTIATE_TEST_SUITE_P(accuracy_2X2p1111, myriadLayers_IR3_BatchPoolingTests_smoke,
                        ::testing::Combine(
                                ::testing::Values<InferenceEngine::SizeVector>({1, 512, 26, 26})
                                , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 2, 2)) /* kernel     */
                                , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 1, 1)) /* stride     */
                                , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 1, 1)) /* pads_begin */
                                , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 1, 1)) /* pads_end   */
                                , ::testing::ValuesIn(s_poolingAutoPad)
                                , ::testing::ValuesIn(s_poolingExcludePad)
                                , ::testing::ValuesIn(s_poolingMethod)
                        )
);
INSTANTIATE_TEST_SUITE_P(accuracy_2X2p1110, myriadLayers_IR3_BatchPoolingTests_smoke,
                        ::testing::Combine(
                                ::testing::Values<InferenceEngine::SizeVector>({1, 512, 26, 26})
                                , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 2, 2)) /* kernel     */
                                , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 1, 1)) /* stride     */
                                , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 1, 1)) /* pads_begin */
                                , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 1, 0)) /* pads_end   */
                                , ::testing::ValuesIn(s_poolingAutoPad)
                                , ::testing::ValuesIn(s_poolingExcludePad)
                                , ::testing::ValuesIn(s_poolingMethod)
                        )
);
INSTANTIATE_TEST_SUITE_P(accuracy_2X2p1100, myriadLayers_IR3_BatchPoolingTests_smoke,
                        ::testing::Combine(
                                ::testing::Values<InferenceEngine::SizeVector>({1, 512, 26, 26})
                                , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 2, 2)) /* kernel     */
                                , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 1, 1)) /* stride     */
                                , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 1, 1)) /* pads_begin */
                                , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 0, 0)) /* pads_end   */
                                , ::testing::ValuesIn(s_poolingAutoPad)
                                , ::testing::ValuesIn(s_poolingExcludePad)
                                , ::testing::ValuesIn(s_poolingMethod)
                        )
);
INSTANTIATE_TEST_SUITE_P(accuracy_2X2p1000, myriadLayers_IR3_BatchPoolingTests_smoke,
                        ::testing::Combine(
                                ::testing::Values<InferenceEngine::SizeVector>({1, 512, 26, 26})
                                , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 2, 2)) /* kernel     */
                                , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 1, 1)) /* stride     */
                                , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 1, 0)) /* pads_begin */
                                , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 0, 0)) /* pads_end   */
                                , ::testing::ValuesIn(s_poolingAutoPad)
                                , ::testing::ValuesIn(s_poolingExcludePad)
                                , ::testing::ValuesIn(s_poolingMethod)
                        )
);
INSTANTIATE_TEST_SUITE_P(accuracy_2X2p1101, myriadLayers_IR3_BatchPoolingTests_smoke,
                        ::testing::Combine(
                                ::testing::Values<InferenceEngine::SizeVector>({1, 512, 26, 26})
                                , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 2, 2)) /* kernel     */
                                , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 1, 1)) /* stride     */
                                , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 1, 1)) /* pads_begin */
                                , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 0, 1)) /* pads_end   */
                                , ::testing::ValuesIn(s_poolingAutoPad)
                                , ::testing::ValuesIn(s_poolingExcludePad)
                                , ::testing::ValuesIn(s_poolingMethod)
                        )
);
INSTANTIATE_TEST_SUITE_P(accuracy_2X2p1011, myriadLayers_IR3_BatchPoolingTests_smoke,
                        ::testing::Combine(
                                ::testing::Values<InferenceEngine::SizeVector>({1, 512, 26, 26})
                                , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 2, 2)) /* kernel     */
                                , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 1, 1)) /* stride     */
                                , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 1, 0)) /* pads_begin */
                                , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 1, 1)) /* pads_end   */
                                , ::testing::ValuesIn(s_poolingAutoPad)
                                , ::testing::ValuesIn(s_poolingExcludePad)
                                , ::testing::ValuesIn(s_poolingMethod)
                        )
);

INSTANTIATE_TEST_SUITE_P(accuracy, myriadLayersTestsMax_smoke,
                        ::testing::Combine(
                                ::testing::ValuesIn(g_poolingInput),
                                ::testing::ValuesIn(g_poolingLayerParamsFull),
                                ::testing::ValuesIn(g_poolingLayout))
);

INSTANTIATE_TEST_SUITE_P(accuracy, myriadLayersTestsMaxOverlappedByKernel_smoke,
                        ::testing::Combine(
                            ::testing::Values<InferenceEngine::SizeVector>({1, 1024, 6, 6}),
                            ::testing::Values<param_size>(MAKE_STRUCT(param_size, 7, 7)),
                            ::testing::Values<param_size>(MAKE_STRUCT(param_size, 1, 1)),
                            ::testing::Values<paddings4>(MAKE_STRUCT(paddings4, 0, 0, 1, 1)),
                            ::testing::ValuesIn(g_poolingLayout))
);

INSTANTIATE_TEST_SUITE_P(accuracy, myriadLayersTestsMaxPad4_smoke,
                        ::testing::Combine(
                                ::testing::ValuesIn(g_poolingInputPad4),
                                ::testing::ValuesIn(g_poolingKernelPad4),
                                ::testing::ValuesIn(g_poolingStridePad4),
                                ::testing::ValuesIn(g_poolingPad4),
                                ::testing::ValuesIn(g_poolingLayout))
);

INSTANTIATE_TEST_SUITE_P(accuracy, myriadLayersTestsAvgPad4_smoke,
                        ::testing::Combine(
                                ::testing::ValuesIn(g_poolingInputPad4),
                                ::testing::ValuesIn(g_poolingKernelPad4),
                                ::testing::ValuesIn(g_poolingStridePad4),
                                ::testing::ValuesIn(g_poolingPad4),
                                ::testing::ValuesIn(g_poolingLayout))
);

INSTANTIATE_TEST_SUITE_P(accuracy, myriadLayersTestsGlobalMax_smoke,
                        ::testing::ValuesIn(g_GlobalPoolingInput ));

INSTANTIATE_TEST_SUITE_P(accuracy_3x3, myriadLayersTestsMax_smoke,
                        ::testing::Combine(
                                ::testing::ValuesIn(g_poolingInput),
                                ::testing::ValuesIn(s_poolingLayerParams_k3x3),
                                ::testing::ValuesIn(g_poolingLayout))
);

INSTANTIATE_TEST_SUITE_P(accuracy, myriadLayersTestsAvg_smoke,
                        ::testing::Combine(
                                ::testing::ValuesIn(g_poolingInput),
                                ::testing::ValuesIn(g_poolingLayerParamsFull),
                                ::testing::ValuesIn(g_poolingLayout))
);

INSTANTIATE_TEST_SUITE_P(accuracy, myriadLayersTestsAvgOverlappedByKernel_smoke,
                        ::testing::Combine(
                            ::testing::Values<InferenceEngine::SizeVector>({1, 1024, 6, 6}),
                            ::testing::Values<param_size>(MAKE_STRUCT(param_size, 7, 7)),
                            ::testing::Values<param_size>(MAKE_STRUCT(param_size, 1, 1)),
                            ::testing::Values<paddings4>(MAKE_STRUCT(paddings4, 0, 0, 1, 1)),
                            ::testing::ValuesIn(g_poolingLayout))
);

INSTANTIATE_TEST_SUITE_P(accuracy_3x3, myriadLayersTestsAvg_smoke,
                        ::testing::Combine(
                                ::testing::ValuesIn(g_poolingInput),
                                ::testing::ValuesIn(s_poolingLayerParams_k3x3),
                                ::testing::ValuesIn(g_poolingLayout))
);

INSTANTIATE_TEST_SUITE_P(accuracy, myriadLayersTestsGlobalAvg_smoke,
                        ::testing::ValuesIn(g_GlobalPoolingInput));
