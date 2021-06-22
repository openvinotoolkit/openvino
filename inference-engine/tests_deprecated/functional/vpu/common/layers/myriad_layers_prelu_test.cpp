// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_prelu_test.hpp"

INSTANTIATE_TEST_SUITE_P(accuracy_PReLU, myriadLayerPReLU_smoke,
    ::testing::Combine(
        ::testing::ValuesIn(s_PReLUTensors)
      , ::testing::Values<ChannelSharedPrelu>(0, 1)
    )
);

INSTANTIATE_TEST_SUITE_P(
    accuracy, myriadLayerFullyConnectedWithPReLU_smoke,
    ::testing::Combine(
        ::testing::ValuesIn(g_fcTestParamsSubset),
        ::testing::Values(g_dimensionsFC[0]),
        ::testing::ValuesIn(g_addBiasFC),
        ::testing::ValuesIn(s_PReluLayerParams)
    )
);

INSTANTIATE_TEST_SUITE_P(accuracy, myriadLayersTestsMaxPoolingWithPReLU_smoke,
    ::testing::Combine(
        ::testing::ValuesIn(g_poolingInput),
        ::testing::ValuesIn(g_poolingLayerParamsLite),
        ::testing::ValuesIn(g_poolingLayout),
        ::testing::ValuesIn(s_PReluLayerParams))
);

INSTANTIATE_TEST_SUITE_P(accuracy, myriadLayersTestsAvgPoolingWithPReLU_smoke,
    ::testing::Combine(
        ::testing::ValuesIn(g_poolingInput),
        ::testing::ValuesIn(g_poolingLayerParamsLite),
        ::testing::ValuesIn(g_poolingLayout),
        ::testing::ValuesIn(s_PReluLayerParams))
);

INSTANTIATE_TEST_SUITE_P(accuracy_postop, myriadLayersTestsMaxPoolingWithPReLU_smoke,
    ::testing::Combine(
        ::testing::ValuesIn(g_poolingInput_postOp),
        ::testing::Values<pooling_layer_params>(MAKE_STRUCT(pooling_layer_params, {3, 3}, {1, 1}, {1, 1})),
        ::testing::ValuesIn(g_poolingLayout),
        ::testing::Values<PReLULayerDef>(MAKE_STRUCT(PReLULayerDef, {{{PRELU_PARAM, "0"}}})))
);

INSTANTIATE_TEST_SUITE_P(accuracy_postop, myriadLayersTestsAvgPoolingWithPReLU_smoke,
    ::testing::Combine(
        ::testing::ValuesIn(g_poolingInput_postOp),
        ::testing::Values<pooling_layer_params>(MAKE_STRUCT(pooling_layer_params, {3, 3}, {1, 1}, {1, 1})),
        ::testing::ValuesIn(g_poolingLayout),
        ::testing::Values<PReLULayerDef>(MAKE_STRUCT(PReLULayerDef, {{{PRELU_PARAM, "0"}}})))
);

INSTANTIATE_TEST_SUITE_P(accuracy, myriadLayerConvolutionWithPReLU_smoke,
        ::testing::Combine(
            ::testing::ValuesIn(g_convolutionTensors)
          , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 3, 3))
          , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<uint32_t>(16)
          , ::testing::Values<uint32_t>(1)
          , ::testing::ValuesIn(s_PReluLayerParams)
          )
);

INSTANTIATE_TEST_SUITE_P(accuracy_postop, myriadLayerConvolutionWithPReLU_smoke,
        ::testing::Combine(
            ::testing::ValuesIn(g_poolingInput_postOp)
          , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 3, 3))
          , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 1, 1)/*, MAKE_STRUCT(param_size, 2, 2)*/)
          , ::testing::Values<param_size>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<uint32_t>(32)
          , ::testing::Values<uint32_t>(1)
          , ::testing::Values<PReLULayerDef>(MAKE_STRUCT(PReLULayerDef, {{{PRELU_PARAM, "0"}}}))
          )
);

