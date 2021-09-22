// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_convolution_test.hpp"

INSTANTIATE_TEST_SUITE_P(accuracy_chw_dilation, myriadLayerConvolution_smoke,
        ::testing::Combine(
            ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 37, 43, 43)
                                       , MAKE_STRUCT(tensor_test_params, 1, 37, 19, 19))
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1)
                                    , MAKE_STRUCT(param_size, 2, 2)
                                    , MAKE_STRUCT(param_size, 3, 3)
                                    )
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 3, 2))
          , ::testing::Values<out_channels>(24)
          , ::testing::Values<group>(1)
          , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 6, 5))
          , ::testing::Values<layoutPreference>(vpu::LayoutPreference::ChannelMajor)
        )
);

INSTANTIATE_TEST_SUITE_P(accuracy, myriadLayers_IR3_ConvTests_smoke,
        ::testing::Combine(
            ::testing::Values<InferenceEngine::SizeVector>({1, 3, 32, 24})
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<uint32_t>(12)
          , ::testing::Values<uint32_t>(1)
          )
);

INSTANTIATE_TEST_SUITE_P(accuracy_Batch_0, myriadLayers_BatchTest_ConvTests_smoke,
        ::testing::Combine(
            ::testing::Values<InferenceEngine::SizeVector>({10, 5, 1, 1})
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
          , ::testing::Values<uint32_t>(5)
          , ::testing::Values<uint32_t>(1)
          )
);

INSTANTIATE_TEST_SUITE_P(accuracy_Batch_1, myriadLayers_BatchTest_ConvTests_smoke,
        ::testing::Combine(
            ::testing::Values<InferenceEngine::SizeVector>({10, 576, 7, 7})
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
          , ::testing::Values<uint32_t>(128)
          , ::testing::Values<uint32_t>(1)
          )
);

INSTANTIATE_TEST_SUITE_P(accuracy_Batch_2, myriadLayers_BatchTest_ConvTests_smoke,
        ::testing::Combine(
            ::testing::Values<InferenceEngine::SizeVector>({10, 128, 7, 7})
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 2, 2))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<uint32_t>(192)
          , ::testing::Values<uint32_t>(1)
          )
);

INSTANTIATE_TEST_SUITE_P(accuracy_Batch_3, myriadLayers_BatchTest_ConvTests_smoke,
        ::testing::Combine(
            ::testing::Values<InferenceEngine::SizeVector>({10, 4, 7, 7})
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
          , ::testing::Values<uint32_t>(4)
          , ::testing::Values<uint32_t>(1)
          )
);

INSTANTIATE_TEST_SUITE_P(accuracy_Batch_4, myriadLayers_BatchTest_ConvTests_smoke,
        ::testing::Combine(
            ::testing::Values<InferenceEngine::SizeVector>({10, 256, 7, 7})
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 2, 2))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<uint32_t>(256)
          , ::testing::Values<uint32_t>(1)
          )
);

INSTANTIATE_TEST_SUITE_P(accuracy_Batch_5, myriadLayers_BatchTest_ConvTests_smoke,
        ::testing::Combine(
            ::testing::Values<InferenceEngine::SizeVector>({10, 1024, 4, 4})
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
          , ::testing::Values<uint32_t>(352)
          , ::testing::Values<uint32_t>(1)
          )
);

INSTANTIATE_TEST_SUITE_P(accuracy_Batch_6, myriadLayers_BatchTest_ConvTests_smoke,
        ::testing::Combine(
            ::testing::Values<InferenceEngine::SizeVector>({10, 192, 4, 4})
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<uint32_t>(320)
          , ::testing::Values<uint32_t>(1)
          )
);

INSTANTIATE_TEST_SUITE_P(accuracy_Batch_7, myriadLayers_BatchTest_ConvTests_smoke,
        ::testing::Combine(
            ::testing::Values<InferenceEngine::SizeVector>({10, 160, 4, 4})
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<uint32_t>(224)
          , ::testing::Values<uint32_t>(1)
          )
);

INSTANTIATE_TEST_SUITE_P(accuracy_Batch_8, myriadLayers_BatchTest_ConvTests_smoke,
        ::testing::Combine(
            ::testing::Values<InferenceEngine::SizeVector>({10, 224, 4, 4})
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<uint32_t>(224)
          , ::testing::Values<uint32_t>(1)
          )
);

INSTANTIATE_TEST_SUITE_P(accuracy_Batch_9, myriadLayers_BatchTest_ConvTests_smoke,
        ::testing::Combine(
            ::testing::Values<InferenceEngine::SizeVector>({10, 1024, 4, 4})
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
          , ::testing::Values<uint32_t>(128)
          , ::testing::Values<uint32_t>(1)
          )
);

INSTANTIATE_TEST_SUITE_P(accuracy_Batch_10, myriadLayers_BatchTest_ConvTests_smoke,
        ::testing::Combine(
            ::testing::Values<InferenceEngine::SizeVector>({1, 64, 56, 56})
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<uint32_t>(192)
          , ::testing::Values<uint32_t>(1)
          )
);

INSTANTIATE_TEST_SUITE_P(accuracy_Batch_11, myriadLayers_BatchTest_ConvTests_smoke,
        ::testing::Combine(
            ::testing::Values<InferenceEngine::SizeVector>({10, 192, 7, 7})
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<uint32_t>(256)
          , ::testing::Values<uint32_t>(1)
          )
);

INSTANTIATE_TEST_SUITE_P(accuracy_Batch_12, myriadLayers_BatchTest_ConvTests_smoke,
        ::testing::Combine(
            ::testing::Values<InferenceEngine::SizeVector>({10, 576, 7, 7})
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
          , ::testing::Values<uint32_t>(192)
          , ::testing::Values<uint32_t>(1)
          )
);

INSTANTIATE_TEST_SUITE_P(accuracy_Batch_1, myriadLayers_BatchTest2_ConvTests_smoke,
        ::testing::Combine(
            ::testing::Values<InferenceEngine::SizeVector>({10, 576, 7, 7})
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
          , ::testing::Values<uint32_t>(192)
          , ::testing::Values<uint32_t>(1)
          )
);

INSTANTIATE_TEST_SUITE_P(accuracy_3X3, myriadLayers_IR3_ConvTests_smoke,
        ::testing::Combine(
            ::testing::Values<InferenceEngine::SizeVector>({1, 3, 32, 24})
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<uint32_t>(12)
          , ::testing::Values<uint32_t>(1)
          )
);

INSTANTIATE_TEST_SUITE_P(accuracy_3X1, myriadLayers_IR3_ConvTests_smoke,
        ::testing::Combine(
            ::testing::Values<InferenceEngine::SizeVector>({1, 3, 32, 24})
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 1))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 0))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 0))
          , ::testing::Values<uint32_t>(16)
          , ::testing::Values<uint32_t>(1)
          )
);

INSTANTIATE_TEST_SUITE_P(accuracy_1X3, myriadLayers_IR3_ConvTests_smoke,
        ::testing::Combine(
            ::testing::Values<InferenceEngine::SizeVector>({1, 4, 16, 16})
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 1, 3))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 1))
          , ::testing::Values<uint32_t>(8)
          , ::testing::Values<uint32_t>(1)
          )
);

INSTANTIATE_TEST_SUITE_P(accuracy, myriadLayers_3X3X3_ConstInput_smoke,
        ::testing::Combine(
            ::testing::Values<InferenceEngine::SizeVector>({1, 3, 10, 10})
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<uint32_t>(32)
          , ::testing::Values<uint32_t>(1)
          , ::testing::ValuesIn(g_poolingLayout) // this array keeps possible layouts
          )
);

INSTANTIATE_TEST_SUITE_P(accuracy_crossroad_spatialConv, myriadLayerConvolutionTensorFlow,
        ::testing::Combine(
            ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 3, 1024, 1024))
          , ::testing::Values<DimsOutput>(MAKE_STRUCT(tensor_test_params, 1, 3, 512, 512))
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 7, 7))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 2, 2))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 3, 3))
          , ::testing::Values<group>(1)
          , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 2))
          , ::testing::Values<layoutPreference>(vpu::LayoutPreference::ChannelMinor)
          )
);

INSTANTIATE_TEST_SUITE_P(accuracy_inception_v2, myriadLayerConvolutionTensorFlow,
        ::testing::Combine(
            ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 16, 28, 28))
          , ::testing::Values<DimsOutput>(MAKE_STRUCT(tensor_test_params, 1, 64, 14, 14))
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 2, 2))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
          , ::testing::Values<group>(1)
          , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 2))
          , ::testing::Values<layoutPreference>(vpu::LayoutPreference::ChannelMinor)
          )
);

INSTANTIATE_TEST_SUITE_P(accuracy_inception_v1, myriadLayerConvolutionTensorFlow,
        ::testing::Combine(
            ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 3, 224, 224),
                                         MAKE_STRUCT(tensor_test_params, 1, 32, 224, 224)
            )
          , ::testing::Values<DimsOutput>(MAKE_STRUCT(tensor_test_params, 1, 64, 112, 112))
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 7, 7))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 2, 2))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 2, 2))
          , ::testing::Values<group>(1)
          , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 2))
          , ::testing::Values<layoutPreference>(vpu::LayoutPreference::ChannelMinor)
          )
);

INSTANTIATE_TEST_SUITE_P(test_3x3_SSD_dilation, myriadLayerConvolution,
        ::testing::Combine(
            ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 512, 19, 19))
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 6, 6))
          , ::testing::Values<out_channels>(1024)
          , ::testing::Values<group>(1)
          , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 6, 5))
          , ::testing::Values<layoutPreference>(vpu::LayoutPreference::ChannelMinor)
          )
);

INSTANTIATE_TEST_SUITE_P(test_TF_Resnet_50, myriadLayers_IR3_ConvTests_smoke,
        ::testing::Combine(
            ::testing::Values<InferenceEngine::SizeVector>({1, 512, 38, 38})
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
          , ::testing::Values<out_channels>(128)
          , ::testing::Values<group>(1)
          )
);

INSTANTIATE_TEST_SUITE_P(test_3x3_icvnet_dilation, myriadLayerConvolution,
        ::testing::Combine(
            ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 24, 20, 20))
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1)
                                    , MAKE_STRUCT(param_size, 2, 2))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1)
                                 , MAKE_STRUCT(param_size, 2, 2))
          , ::testing::Values<out_channels>(48)
          , ::testing::Values<group>(6, 8)
          , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 2, 3))
          , ::testing::Values<layoutPreference>(vpu::LayoutPreference::ChannelMinor)
          )
);

INSTANTIATE_TEST_SUITE_P(test_5x5_with_dilation, myriadLayerConvolution,
        ::testing::Combine(
            ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 32, 64,  77))
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 5, 5))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1)
                                    , MAKE_STRUCT(param_size, 2, 2))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1)
                                 , MAKE_STRUCT(param_size, 0, 0)
                                 , MAKE_STRUCT(param_size, 2, 2))
          , ::testing::Values<out_channels>(32)
          , ::testing::Values<group>(1)
          , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 2, 3), MAKE_STRUCT(param_size, 3, 4))
          , ::testing::Values<layoutPreference>(vpu::LayoutPreference::ChannelMinor)
          )
);

INSTANTIATE_TEST_SUITE_P(test_7x7_with_dilation, myriadLayerConvolution,
        ::testing::Combine(
            ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 32, 64,  77))
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 7, 7))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1)
                                    , MAKE_STRUCT(param_size, 2, 2))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1)
                                 , MAKE_STRUCT(param_size, 0, 0)
                                 , MAKE_STRUCT(param_size, 2, 2))
          , ::testing::Values<out_channels>(32)
          , ::testing::Values<group>(1)
          , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 2, 3), MAKE_STRUCT(param_size, 3, 4))
          , ::testing::Values<layoutPreference>(vpu::LayoutPreference::ChannelMinor)
          )
);


INSTANTIATE_TEST_SUITE_P(test_conv1x1, myriadLayerConvolution,
        ::testing::Combine(
            ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 10, 13, 13))
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1)
                                    , MAKE_STRUCT(param_size, 2, 2))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<out_channels>(20)
          , ::testing::Values<group>(1)
          , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 2))
          , ::testing::Values<layoutPreference>(vpu::LayoutPreference::ChannelMinor)
           )
);

INSTANTIATE_TEST_SUITE_P(test_yolo_tiny_2_512x13x13_use_3x3_convolution, myriadLayerConvolution,
        ::testing::Combine(
            ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 512, 13, 13))
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<out_channels>(1024)
          , ::testing::Values<group>(1)
          , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 2))
          , ::testing::Values<layoutPreference>(vpu::LayoutPreference::ChannelMinor)
           )
);

INSTANTIATE_TEST_SUITE_P(test_yolo_tiny_2_512x13x13_use_1x1_convolution, myriadLayerConvolution,
        ::testing::Combine(
            ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 4608, 13, 13))
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
          , ::testing::Values<out_channels>(1024)
          , ::testing::Values<group>(1)
          , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 2))
          , ::testing::Values<layoutPreference>(vpu::LayoutPreference::ChannelMinor)
           )
);

INSTANTIATE_TEST_SUITE_P(accuracy_group, myriadLayerConvolution,
        ::testing::Combine(
            ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 32, 64,  77)
                                       , MAKE_STRUCT(tensor_test_params, 1, 32, 112, 96))
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3)
                                  , MAKE_STRUCT(param_size, 5, 5)
                                  , MAKE_STRUCT(param_size, 7, 7))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1)
                                    , MAKE_STRUCT(param_size, 2, 2))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1)
                                 , MAKE_STRUCT(param_size, 0, 0)
                                 , MAKE_STRUCT(param_size, 2, 2))
          , ::testing::Values<out_channels>(32)
          , ::testing::Values<group>(32)
          , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 2))
          , ::testing::Values<layoutPreference>(vpu::LayoutPreference::ChannelMinor)
          )
);

INSTANTIATE_TEST_SUITE_P(accuracy_group_large_input, myriadLayerConvolution,
        ::testing::Combine(
            ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 32, 192, 336))
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1)
                                 , MAKE_STRUCT(param_size, 0, 0)
                                 , MAKE_STRUCT(param_size, 2, 2))
          , ::testing::Values<out_channels>(32)
          , ::testing::Values<group>(32)
          , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 2))
          , ::testing::Values<layoutPreference>(vpu::LayoutPreference::ChannelMinor)
          )
);

INSTANTIATE_TEST_SUITE_P(accuracy_any_group, myriadLayerConvolution_smoke,
        ::testing::Combine(
            ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 32, 64,  77))
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1)
                                    , MAKE_STRUCT(param_size, 2, 2))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1)
                                 , MAKE_STRUCT(param_size, 0, 0)
                                 , MAKE_STRUCT(param_size, 2, 2))
          , ::testing::Values<out_channels>(32)
          , ::testing::Values<group>(2, 4, 16)
          , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 2), MAKE_STRUCT(param_size, 2, 3), MAKE_STRUCT(param_size, 3, 4)),
            ::testing::Values<layoutPreference>(vpu::LayoutPreference::ChannelMajor)
        )
);

INSTANTIATE_TEST_SUITE_P(accuracy_any_group, myriadLayerConvolution,
        ::testing::Combine(
            ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 32, 64,  77))
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1)
                                    , MAKE_STRUCT(param_size, 2, 2))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1)
                                 , MAKE_STRUCT(param_size, 0, 0)
                                 , MAKE_STRUCT(param_size, 2, 2))
          , ::testing::Values<out_channels>(32)
          , ::testing::Values<group>(2, 4, 16)
          , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 2), MAKE_STRUCT(param_size, 2, 3), MAKE_STRUCT(param_size, 3, 4)),
            ::testing::Values<layoutPreference>(vpu::LayoutPreference::ChannelMinor)
        )
);

INSTANTIATE_TEST_SUITE_P(set_optimization_for_3x3_with_group, myriadLayerConvolution,
        ::testing::Combine(
            ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 24, 80, 80)
                                       , MAKE_STRUCT(tensor_test_params, 1, 36, 80, 80))
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1),
                                      MAKE_STRUCT(param_size, 2, 2))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<out_channels>(24)
          , ::testing::Values<group>(6)
          , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 2), MAKE_STRUCT(param_size, 2, 3))
          , ::testing::Values<layoutPreference>(vpu::LayoutPreference::ChannelMinor)
          )
);

INSTANTIATE_TEST_SUITE_P(set_optimization_for_3x3s1, myriadLayerConvolution,
        ::testing::Combine(
            ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 24, 80, 80))
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<out_channels>(1,2,3,4,5,6,7,8)
          , ::testing::Values<group>(1)
          , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 2))
          , ::testing::Values<layoutPreference>(vpu::LayoutPreference::ChannelMinor)
          )
);

INSTANTIATE_TEST_SUITE_P(accuracy_1x1, myriadLayerConvolution,
        ::testing::Combine(
            ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 16, 64, 64)
                                       , MAKE_STRUCT(tensor_test_params, 1, 32, 1, 1))
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1)
                                    , MAKE_STRUCT(param_size, 2, 2))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0),
                                   MAKE_STRUCT(param_size, 1, 1)
                                  )
          , ::testing::Values<out_channels>(16, 24)
          , ::testing::Values<group>(1)
          , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 2))
          , ::testing::Values<layoutPreference>(vpu::LayoutPreference::ChannelMinor)
          )
);

INSTANTIATE_TEST_SUITE_P(accuracy_3x3, myriadLayerConvolution,
        ::testing::Combine(
            ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 8, 16, 16)
                                       , MAKE_STRUCT(tensor_test_params, 1, 8, 59, 73))
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1)
                                    , MAKE_STRUCT(param_size, 2, 2))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0)
                                 , MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<out_channels>(8, 15/*, 20 failed for 3x3s2*/, 32)
          , ::testing::Values<group>(1)
          , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 2))
          , ::testing::Values<layoutPreference>(vpu::LayoutPreference::ChannelMinor)
          )
);

INSTANTIATE_TEST_SUITE_P(accuracy_1x3, myriadLayerConvolution,
        ::testing::Combine(
            ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 8, 59, 73))
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 1, 3), MAKE_STRUCT(param_size, 3, 1))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0)
                                 , MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<out_channels>(7, 32)
          , ::testing::Values<group>(1)
          , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 2))
          , ::testing::Values<layoutPreference>(vpu::LayoutPreference::ChannelMinor)
          )
);

INSTANTIATE_TEST_SUITE_P(accuracy_5x5, myriadLayerConvolution_smoke,
        ::testing::Combine(
            ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 16, 32, 32)
                                     /*, MAKE_STRUCT(tensor_test_params, 1, 8, 511, 399) failed*/)
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 5, 5))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1)
                                    , MAKE_STRUCT(param_size, 2, 2))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0)
                               /*, MAKE_STRUCT(param_size, 1, 1) failed*/
                                 , MAKE_STRUCT(param_size, 2, 2))
          , ::testing::Values<out_channels>(16, 32)
          , ::testing::Values<group>(1)
          , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 2)),
            ::testing::Values<layoutPreference>(vpu::LayoutPreference::ChannelMajor)
        )
);

INSTANTIATE_TEST_SUITE_P(accuracy_5x5, myriadLayerConvolution,
        ::testing::Combine(
            ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 16, 32, 32)
                                     /*, MAKE_STRUCT(tensor_test_params, 1, 8, 511, 399) failed*/)
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 5, 5))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1)
                                    , MAKE_STRUCT(param_size, 2, 2))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0)
                               /*, MAKE_STRUCT(param_size, 1, 1) failed*/
                                 , MAKE_STRUCT(param_size, 2, 2))
          , ::testing::Values<out_channels>(16, 32)
          , ::testing::Values<group>(1)
          , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 2)),
            ::testing::Values<layoutPreference>(vpu::LayoutPreference::ChannelMinor)
        )
);

INSTANTIATE_TEST_SUITE_P(accuracy_7x7, myriadLayerConvolution,
        ::testing::Combine(
            ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 8, 32, 32)
                                     /*, MAKE_STRUCT(tensor_test_params, 1, 8, 511, 399) failed*/)
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 7, 7))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1)
                                    , MAKE_STRUCT(param_size, 2, 2))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0)
                               /*, MAKE_STRUCT(param_size, 1, 1) failed*/
                               /*, MAKE_STRUCT(param_size, 3, 3) failed*/)
          , ::testing::Values<out_channels>(16, 32)
          , ::testing::Values<group>(1)
          , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 2))
          , ::testing::Values<layoutPreference>(vpu::LayoutPreference::ChannelMinor)
          )
);

INSTANTIATE_TEST_SUITE_P(accuracy_3x3_large_input_1, myriadLayerConvolution,
        ::testing::Combine(
            ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 3, 720, 1280))
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 2, 2))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
          , ::testing::Values<out_channels>(8)
          , ::testing::Values<group>(1)
          , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 2))
          , ::testing::Values<layoutPreference>(vpu::LayoutPreference::ChannelMinor)
          )
);

INSTANTIATE_TEST_SUITE_P(accuracy_3x3_large_input_2, myriadLayerConvolution,
        ::testing::Combine(
            ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 24, 357, 637))
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
          , ::testing::Values<out_channels>(8)
          , ::testing::Values<group>(1)
          , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 2))
          , ::testing::Values<layoutPreference>(vpu::LayoutPreference::ChannelMinor)
          )
);


INSTANTIATE_TEST_SUITE_P(accuracy_3x3_large_input_3, myriadLayerConvolution,
        ::testing::Combine(
            ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 16, 359, 639))
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
          , ::testing::Values<out_channels>(12)
          , ::testing::Values<group>(1)
          , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 2))
          , ::testing::Values<layoutPreference>(vpu::LayoutPreference::ChannelMinor)
          )
);

INSTANTIATE_TEST_SUITE_P(accuracy_1x1_large_input, myriadLayerConvolution,
        ::testing::Combine(
            ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 24, 355, 635))
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
          , ::testing::Values<out_channels>(2, 3)
          , ::testing::Values<group>(1)
          , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 2))
          , ::testing::Values<layoutPreference>(vpu::LayoutPreference::ChannelMinor)
          )
);

INSTANTIATE_TEST_SUITE_P(accuracy_small_input_0, myriadLayerConvolution,
        ::testing::Combine(
            ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 128, 38, 38))
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<out_channels>(6)
          , ::testing::Values<group>(1)
          , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 2))
          , ::testing::Values<layoutPreference>(vpu::LayoutPreference::ChannelMinor)
          )
);
INSTANTIATE_TEST_SUITE_P(accuracy_small_input_1, myriadLayerConvolution,
        ::testing::Combine(
            ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 256, 2, 3))
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<out_channels>(8)
          , ::testing::Values<group>(1)
          , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<layoutPreference>(vpu::LayoutPreference::ChannelMinor)
          )
);
INSTANTIATE_TEST_SUITE_P(accuracy_small_input_2, myriadLayerConvolution,
        ::testing::Combine(
            ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 256, 2, 2))
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<out_channels>(8)
          , ::testing::Values<group>(1)
          , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<layoutPreference>(vpu::LayoutPreference::ChannelMinor)
          )
);
INSTANTIATE_TEST_SUITE_P(accuracy_small_input_3, myriadLayerConvolution,
        ::testing::Combine(
            ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 256, 1, 1))
          , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
          , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<out_channels>(84)
          , ::testing::Values<group>(1)
          , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 1))
          , ::testing::Values<layoutPreference>(vpu::LayoutPreference::ChannelMinor)
           )
 );

// TODO: rewrite to ngraph to have reshape functionality
TEST_F(myriadLayersTests_nightly, DISABLED_tests125) {
    std::string outName1 = "SecondStageFeatureExtractor/InceptionV2/Mixed_5a/Branch_1/Conv2d_0b_3x3/Conv2D";
    std::string outName2 = "SecondStageFeatureExtractor/InceptionV2/Mixed_5a/Branch_0/Conv2d_1a_3x3/Relu";
    InferenceEngine::TBlob<uint8_t>::Ptr weights(GenWeights(1697280 / sizeof(ie_fp16)));
    constWeightsRange(weights->data().as<uint16_t *>(), 1697280 / sizeof(ie_fp16));

    InferenceEngine::Core ie;
    auto network = ie.ReadNetwork(MODEL_RFCNN, weights);

    auto inputsInfo = network.getInputsInfo();
    inputsInfo["input"]->setPrecision(Precision::FP16);

    auto outputsInfo = network.getOutputsInfo();
    outputsInfo[outName1]->setPrecision(Precision::FP16);
    outputsInfo[outName2]->setPrecision(Precision::FP16);

    InferenceEngine::ExecutableNetwork exeNetwork;
    ASSERT_NO_THROW(exeNetwork = _vpuPluginPtr->LoadNetwork(network));

    InferenceEngine::InferRequest     inferRequest;
    ASSERT_NO_THROW(inferRequest = exeNetwork.CreateInferRequest());

    Blob::Ptr input;
    ASSERT_NO_THROW(input = inferRequest.GetBlob("input"));
    genTestData(input);

    ASSERT_NO_THROW(inferRequest.Infer());

    Blob::Ptr out1, out2;
    ASSERT_NO_THROW(out1 = inferRequest.GetBlob(outName1.c_str()));
    ASSERT_NO_THROW(out2 = inferRequest.GetBlob(outName2.c_str()));
};

// This tests checks that conv3x3s1 case doesn't corrupt its input.
// To check this we run two convolutions on the same input
// and check that they return same results.
TEST_F(myriadLayersTests_nightly, SmallConv_CorruptInputBug) {
    const std::string model = R"V0G0N(
        <Net name="SmallConv_CorruptInputBug" version="2" batch="1">
            <layers>
                <layer name="input" type="Input" precision="FP16" id="1">
                    <output>
                        <port id="1">
                            <dim>1</dim>
                            <dim>256</dim>
                            <dim>1</dim>
                            <dim>1</dim>
                        </port>
                    </output>
                </layer>
                <layer name="input_copy" type="Power" precision="FP16" id="2">
                    <power_data power="1" scale="1" shift="0"/>
                    <input>
                        <port id="2">
                            <dim>1</dim>
                            <dim>256</dim>
                            <dim>1</dim>
                            <dim>1</dim>
                        </port>
                    </input>
                    <output>
                        <port id="3">
                            <dim>1</dim>
                            <dim>256</dim>
                            <dim>1</dim>
                            <dim>1</dim>
                        </port>
                    </output>
                </layer>
                <layer name="conv1" type="Convolution" precision="FP16" id="3">
                    <convolution_data stride-x="1" stride-y="1" pad-x="1" pad-y="1" kernel-x="3" kernel-y="3" output="84" group="1"/>
                    <input>
                        <port id="4">
                            <dim>1</dim>
                            <dim>256</dim>
                            <dim>1</dim>
                            <dim>1</dim>
                        </port>
                    </input>
                    <output>
                        <port id="5">
                            <dim>1</dim>
                            <dim>84</dim>
                            <dim>1</dim>
                            <dim>1</dim>
                        </port>
                    </output>
                    <weights offset="0" size="387072"/>
                    <biases offset="387072" size="168"/>
                </layer>
                <layer name="conv2" type="Convolution" precision="FP16" id="4">
                    <convolution_data stride-x="1" stride-y="1" pad-x="1" pad-y="1" kernel-x="3" kernel-y="3" output="84" group="1"/>
                    <input>
                        <port id="6">
                            <dim>1</dim>
                            <dim>256</dim>
                            <dim>1</dim>
                            <dim>1</dim>
                        </port>
                    </input>
                    <output>
                        <port id="7">
                            <dim>1</dim>
                            <dim>84</dim>
                            <dim>1</dim>
                            <dim>1</dim>
                        </port>
                    </output>
                    <weights offset="0" size="387072"/>
                    <biases offset="387072" size="168"/>
                </layer>
                <layer name="conv1_out" type="Power" precision="FP16" id="5">
                    <power_data power="1" scale="1" shift="0"/>
                    <input>
                        <port id="8">
                            <dim>1</dim>
                            <dim>84</dim>
                            <dim>1</dim>
                            <dim>1</dim>
                        </port>
                    </input>
                    <output>
                        <port id="9">
                            <dim>1</dim>
                            <dim>84</dim>
                            <dim>1</dim>
                            <dim>1</dim>
                        </port>
                    </output>
                </layer>
                <layer name="conv2_out" type="Power" precision="FP16" id="6">
                    <power_data power="1" scale="1" shift="0"/>
                    <input>
                        <port id="10">
                            <dim>1</dim>
                            <dim>84</dim>
                            <dim>1</dim>
                            <dim>1</dim>
                        </port>
                    </input>
                    <output>
                        <port id="11">
                            <dim>1</dim>
                            <dim>84</dim>
                            <dim>1</dim>
                            <dim>1</dim>
                        </port>
                    </output>
                </layer>
            </layers>
            <edges>
                <edge from-layer="1" from-port="1" to-layer="2" to-port="2"/>
                <edge from-layer="2" from-port="3" to-layer="3" to-port="4"/>
                <edge from-layer="2" from-port="3" to-layer="4" to-port="6"/>
                <edge from-layer="3" from-port="5" to-layer="5" to-port="8"/>
                <edge from-layer="4" from-port="7" to-layer="6" to-port="10"/>
            </edges>
        </Net>
    )V0G0N";

    SetSeed(DEFAULT_SEED_VALUE);

    size_t num_weights = 193536;
    size_t num_bias = 84;

    TBlob<uint8_t>::Ptr weightsBlob(GenWeights(num_weights + num_bias));
    const ie_fp16 *weights = weightsBlob->readOnly().as<const ie_fp16 *>();
    const ie_fp16 *bias = weights + num_weights;

    ASSERT_NO_THROW(readNetwork(model, weightsBlob));

    const auto& network = _cnnNetwork;

    _inputsInfo = network.getInputsInfo();
    _inputsInfo["input"]->setPrecision(Precision::FP16);

    _outputsInfo = network.getOutputsInfo();
    _outputsInfo["conv1_out"]->setPrecision(Precision::FP16);
    _outputsInfo["conv2_out"]->setPrecision(Precision::FP16);

    ASSERT_NO_THROW(_exeNetwork = _vpuPluginPtr->LoadNetwork(network));

    ASSERT_NO_THROW(_inferRequest = _exeNetwork.CreateInferRequest());

    Blob::Ptr input;
    ASSERT_NO_THROW(input = _inferRequest.GetBlob("input"));
    {
        ie_fp16 *dst = input->buffer().as<ie_fp16 *>();
        for (int i = 0; i < input->size(); ++i) {
            float val = static_cast<float>(std::rand()) / RAND_MAX;
            dst[i] = PrecisionUtils::f32tof16(val);
        }
    }

    ASSERT_NO_THROW(_inferRequest.Infer());

    Blob::Ptr conv1;
    ASSERT_NO_THROW(conv1 = _inferRequest.GetBlob("conv1_out"));

    Blob::Ptr conv2;
    ASSERT_NO_THROW(conv2 = _inferRequest.GetBlob("conv2_out"));

    {
        SCOPED_TRACE("CompareCommonAbsolute with itself");
        CompareCommonAbsolute(conv1, conv2, 0.0);
    }

    {
        SCOPED_TRACE("CompareCommonAbsolute with reference");

        _refBlob = make_shared_blob<ie_fp16>({Precision::FP16, conv1->getTensorDesc().getDims(), Layout::NHWC});
        _refBlob->allocate();
        ref_convolution(input, _refBlob, weights, bias, {3, 3}, {1, 1}, {1, 1}, 1);

        CompareCommonAbsolute(conv1, _refBlob, 0.1);
        CompareCommonAbsolute(conv2, _refBlob, 0.1);
    }
}
