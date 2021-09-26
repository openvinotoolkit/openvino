// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_deconvolution_test.hpp"

INSTANTIATE_TEST_SUITE_P(accuracy_deconv_to_conv, myriadLayerDeconvolution_smoke,
        ::testing::Combine(
            ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 6, 5, 6))
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 1), MAKE_STRUCT(param_size, 3, 3))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 0), MAKE_STRUCT(param_size, 0, 1), MAKE_STRUCT(param_size, 0, 0))
          , ::testing::Values<out_channels>(4)
          , ::testing::Values<group>(1)
          , ::testing::Values<layoutPreference>(vpu::LayoutPreference::ChannelMajor)
          , ::testing::Values<hw_optimization>(true)
          )
);

INSTANTIATE_TEST_SUITE_P(accuracy_deconv_to_conv_2, myriadLayerDeconvolution_smoke,
        ::testing::Combine(
            ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 2, 256, 14, 14))
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 2, 2), MAKE_STRUCT(param_size, 3, 3))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 2, 2))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 0), MAKE_STRUCT(param_size, 0, 1), MAKE_STRUCT(param_size, 0, 0))
          , ::testing::Values<out_channels>(256)
          , ::testing::Values<group>(1)
          , ::testing::Values<layoutPreference>(vpu::LayoutPreference::ChannelMajor)
          , ::testing::Values<hw_optimization>(true)
          )
);

INSTANTIATE_TEST_SUITE_P(accuracy_group, myriadLayerDeconvolution_smoke,
        ::testing::Combine(
            ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 384, 4, 2))
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 2, 2)
                                    , MAKE_STRUCT(param_size, 3, 3)
                                    , MAKE_STRUCT(param_size, 3, 4)
                                    , MAKE_STRUCT(param_size, 4, 4)
                                     )
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 2, 2))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<out_channels>(384)
          , ::testing::Values<group>(2, 4)
          , ::testing::Values<layoutPreference>(vpu::LayoutPreference::ChannelMinor, vpu::LayoutPreference::ChannelMajor)
          , ::testing::Values<hw_optimization>(false)
          )
);

INSTANTIATE_TEST_SUITE_P(accuracy_depthDeconv, myriadLayerDeconvolution_smoke,
        ::testing::Combine(
            ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 384, 4, 2))
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 2, 2)
                                    , MAKE_STRUCT(param_size, 3, 3)
                                    , MAKE_STRUCT(param_size, 3, 4)
                                    , MAKE_STRUCT(param_size, 4, 4)
                                     )
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 2, 2))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<out_channels>(384)
          , ::testing::Values<group>(384)
          , ::testing::Values<layoutPreference>(vpu::LayoutPreference::ChannelMajor,
                                                vpu::LayoutPreference::ChannelMinor)
          , ::testing::Values<hw_optimization>(false)
          )
);

INSTANTIATE_TEST_SUITE_P(accuracy, myriadLayerDeconvolution_asymm_pad,
        ::testing::Combine(
            ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 128, 60, 80),
                                         MAKE_STRUCT(tensor_test_params, 1,   2, 37, 59))
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3)
                                     )
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 2, 2)
                                     )
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 0), MAKE_STRUCT(param_size, 2, 2))
          , ::testing::Values<pad_end>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<out_channels>(21)
          , ::testing::Values<group>(1)
          , ::testing::Values<layoutPreference>(
                                                vpu::LayoutPreference::ChannelMajor,
                                                vpu::LayoutPreference::ChannelMinor)
          , ::testing::Values<hw_optimization>(false)
          )
);

INSTANTIATE_TEST_SUITE_P(accuracy, myriadLayerDeconvolution_smoke,
        ::testing::Combine(
            ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 2, 37, 59)
                                       , MAKE_STRUCT(tensor_test_params, 1, 21, 16, 16)
                                       , MAKE_STRUCT(tensor_test_params, 1, 512, 11, 13)
                                         )
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 2, 2)
                                    , MAKE_STRUCT(param_size, 3, 3)
                                    , MAKE_STRUCT(param_size, 4, 4)
                                    , MAKE_STRUCT(param_size, 5, 5)
                                      )
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1)
                                    , MAKE_STRUCT(param_size, 2, 2)
                                      )
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0)
                                 , MAKE_STRUCT(param_size, 1, 1)
                                 , MAKE_STRUCT(param_size, 2, 2)
                                  )
          , ::testing::Values<out_channels>(1, 21)
          , ::testing::Values<group>(1)
          , ::testing::Values<layoutPreference>(vpu::LayoutPreference::ChannelMinor, vpu::LayoutPreference::ChannelMajor)
          , ::testing::Values<hw_optimization>(false)
          )
);

INSTANTIATE_TEST_SUITE_P(extra3x3s1, myriadLayerDeconvolution_smoke,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 256, 1, 1))
                              , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
                              , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
                              , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
                              , ::testing::Values<out_channels>(256)
                              , ::testing::Values<group>(1)
                              , ::testing::Values<layoutPreference>(vpu::LayoutPreference::ChannelMinor)
                              , ::testing::Values<hw_optimization>(false)
                              )
);
