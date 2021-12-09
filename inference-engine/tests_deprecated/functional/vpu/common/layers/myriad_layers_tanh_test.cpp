// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_tanh_test.hpp"

INSTANTIATE_TEST_SUITE_P(
        accuracy, myriadLayersTestsTanh_smoke,
        ::testing::ValuesIn(s_tanhParams));

INSTANTIATE_TEST_SUITE_P(accuracy, myriadLayerConvolutionWithTanH_smoke,
        ::testing::Combine(
            ::testing::ValuesIn(g_convolutionTensors)
          , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 3, 3))
          , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<uint32_t>(16)
          , ::testing::Values<uint32_t>(1)
          , ::testing::Values<IRVersion>(IRVersion::v7, IRVersion::v10)
          )
);

INSTANTIATE_TEST_SUITE_P(accuracy, myriadLayersTestsMaxPoolingWithTanh_smoke,
    ::testing::Combine(
        ::testing::ValuesIn(g_poolingInput),
        ::testing::ValuesIn(g_poolingLayerParamsLite),
        ::testing::ValuesIn(g_poolingLayout))
);

INSTANTIATE_TEST_SUITE_P(accuracy, myriadLayersTestsAvgPoolingWithTanh_smoke,
    ::testing::Combine(
        ::testing::ValuesIn(g_poolingInput),
        ::testing::ValuesIn(g_poolingLayerParamsLite),
        ::testing::ValuesIn(g_poolingLayout))
);

INSTANTIATE_TEST_SUITE_P(
    accuracy, myriadLayerFullyConnectedWithTanH_smoke,
    ::testing::Combine(
        ::testing::ValuesIn(g_fcTestParamsSubset),
        ::testing::Values(g_dimensionsFC[0]),
        ::testing::ValuesIn(g_addBiasFC)
    )
);
