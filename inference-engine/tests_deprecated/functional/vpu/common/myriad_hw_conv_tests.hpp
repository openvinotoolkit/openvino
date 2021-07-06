// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "myriad_hw_tests_base.hpp"

using HWConvParams = std::tuple<DimsInput, kernel, stride, pad, out_channels, group, dilation_factor>;

class MyriadX_HW_Convolution_Tests_nightly
        : public MyriadX_HW_Tests_nightly,
          public testing::WithParamInterface<HWConvParams> {
public:
    tensor_test_params in_dims;
    param_size kernel;
    param_size stride;
    param_size pad;
    size_t out_c;
    size_t group;
    param_size dilation_factor;

    tensor_test_params out_dims;

    IN_OUT_desc in_tensor, out_tensor;

    size_t numWeights;
    size_t numBiases;

    void SetUp() override {
        ASSERT_NO_FATAL_FAILURE(MyriadX_HW_Tests_nightly::SetUp());

        in_dims = std::get<0>(GetParam());
        kernel = std::get<1>(GetParam());
        stride = std::get<2>(GetParam());
        pad = std::get<3>(GetParam());
        out_c = std::get<4>(GetParam());
        group = std::get<5>(GetParam());
        dilation_factor = std::get<6>(GetParam());

        size_t out_w = (in_dims.w + 2 * pad.x - dilation_factor.x * (kernel.x - 1) - 1 + stride.x) / stride.x;
        size_t out_h = (in_dims.h + 2 * pad.y - dilation_factor.y * (kernel.y - 1) - 1 + stride.y) / stride.y;

        out_dims = {1, out_c, out_h, out_w};

        in_tensor.push_back({in_dims.n, in_dims.c, in_dims.h, in_dims.w});
        out_tensor.push_back({out_dims.n, out_dims.c, out_dims.h, out_dims.w});

        numWeights = kernel.x * kernel.y * (in_dims.c / group) * out_dims.c;
        numBiases = out_dims.c;

        _config[InferenceEngine::MYRIAD_HW_DILATION] = CONFIG_VALUE(YES);
    }

    void AddInitialCopyLayer() {
        _testNet.addLayer(LayerInitParams("Copy").in({in_tensor}).out({in_tensor}),
                 ref_copy_wrap);
    }

    void AddConvolutionLayer() {
        std::map<std::string, std::string> convParams = {
                  {"kernel-x", std::to_string(kernel.x)}
                , {"kernel-y", std::to_string(kernel.y)}
                , {"stride-x", std::to_string(stride.x)}
                , {"stride-y", std::to_string(stride.y)}
                , {"pad-x", std::to_string(pad.x)}
                , {"pad-y", std::to_string(pad.y)}
                , {"output", std::to_string(out_c)}
                , {"group", std::to_string(group)}
                , {"dilation-x", std::to_string(dilation_factor.x)}
                , {"dilation-y", std::to_string(dilation_factor.y)}
        };
        _testNet.addLayer(LayerInitParams("Convolution")
                 .params(convParams)
                 .weights(numWeights).fillWeights(defaultWeightsRange)
                 .biases(numBiases).fillBiases(defaultWeightsRange)
                 .in(in_tensor)
                 .out(out_tensor),
                 ref_convolution_wrap);
    }

    void AddReLULayer(float negativeSlope = 0.0) {
        ParamsStruct reluParams = {
            {"negative_slope", std::to_string(negativeSlope)}
        };
        _testNet.addLayer(LayerInitParams("ReLU")
                 .params(reluParams)
                 .in(out_tensor)
                 .out(out_tensor),
                 ref_ReLU_wrap);
    }

    void AddClampLayer(float min = 0.0, float max = 6.0) {
        ParamsStruct clampParams = {
                {"max", std::to_string(max)}
              , {"min", std::to_string(min)}
        };
        _testNet.addLayer(LayerInitParams("Clamp")
                 .params(clampParams)
                 .in(out_tensor)
                 .out(out_tensor),
                 ref_Clamp_wrap);
    }
};

TEST_P(MyriadX_HW_Convolution_Tests_nightly, Single) {
    if (!CheckMyriadX()) {
        GTEST_SKIP() << "Non-MyriadX device";
    }

    AddInitialCopyLayer();
    AddConvolutionLayer();

    float maxerr = 0.002 * (in_dims.c / group) * kernel.x * kernel.y;
    CompareWithSW(maxerr);
}

TEST_P(MyriadX_HW_Convolution_Tests_nightly, WithReLU) {
    if (!CheckMyriadX()) {
        GTEST_SKIP() << "Non-MyriadX device";
    }

    AddInitialCopyLayer();
    AddConvolutionLayer();
    AddReLULayer(0.0f);

    float maxerr = 0.002 * (in_dims.c / group) * kernel.x * kernel.y;
    CompareWithSW(maxerr);
}

TEST_P(MyriadX_HW_Convolution_Tests_nightly, WithLeakyReLU) {
    if (!CheckMyriadX()) {
        GTEST_SKIP() << "Non-MyriadX device";
    }

    AddConvolutionLayer();
    AddReLULayer(0.1f);

    float maxerr = 0.1 * (in_dims.c / group) * kernel.x * kernel.y;
    CompareWithSW(maxerr);
}

TEST_P(MyriadX_HW_Convolution_Tests_nightly, WithClamp) {
    if (!CheckMyriadX()) {
        GTEST_SKIP() << "Non-MyriadX device";
    }

    AddConvolutionLayer();
    AddClampLayer();

    float maxerr = 0.1 * (in_dims.c / group) * kernel.x * kernel.y;
    CompareWithSW(maxerr);
}

TEST_P(MyriadX_HW_Convolution_Tests_nightly, MultipleInfer) {
    if (!CheckMyriadX()) {
        GTEST_SKIP() << "Non-MyriadX device";
    }

    AddConvolutionLayer();

    CompareWithItself(100);
}

INSTANTIATE_TEST_SUITE_P(MaskRcnn101_DILATION, MyriadX_HW_Convolution_Tests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 256, 50, 50),
                                                             MAKE_STRUCT(tensor_test_params, 1, 256, 100, 171))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 2, 2))
                                , ::testing::Values<out_channels>(256)
                                , ::testing::Values<group>(1)
                                , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 2, 2))
                        )
);

INSTANTIATE_TEST_SUITE_P(kernel_7x7_DILATION, MyriadX_HW_Convolution_Tests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 32, 90, 90))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 7, 7))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
                                , ::testing::Values<out_channels>(32)
                                , ::testing::Values<group>(1)
                                , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 4, 4))
                        )
);

INSTANTIATE_TEST_SUITE_P(Unequal_hw_pad_dilationfactor_DILATION, MyriadX_HW_Convolution_Tests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 32, 128, 128))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3),
                                                            MAKE_STRUCT(param_size, 3, 1))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 2, 0))
                                , ::testing::Values<out_channels>(64)
                                , ::testing::Values<group>(1)
                                , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 2, 2),
                                                                     MAKE_STRUCT(param_size, 2, 1))
                        )
);

INSTANTIATE_TEST_SUITE_P(Strides_DILATION, MyriadX_HW_Convolution_Tests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 32, 64, 64))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 2, 2),
                                                            MAKE_STRUCT(param_size, 4, 4))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 4, 4))
                                , ::testing::Values<out_channels>(32)
                                , ::testing::Values<group>(1)
                                , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 2, 2),
                                                                     MAKE_STRUCT(param_size, 4, 4))
                        )
);

INSTANTIATE_TEST_SUITE_P(conv_1x1s1p0_extra1, MyriadX_HW_Convolution_Tests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 64, 180, 320))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
                                , ::testing::Values<out_channels>(64)
                                , ::testing::Values<group>(1)
                                , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 1))
                        )
);

INSTANTIATE_TEST_SUITE_P(conv_1x1s1p0_extra2, MyriadX_HW_Convolution_Tests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 512, 45, 80))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
                                , ::testing::Values<out_channels>(18)
                                , ::testing::Values<group>(1)
                                , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 1))
                        )
);

INSTANTIATE_TEST_SUITE_P(conv_1x1s1p0_extra3, MyriadX_HW_Convolution_Tests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 256, 45, 80))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
                                , ::testing::Values<out_channels>(294)
                                , ::testing::Values<group>(1)
                                , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 1))
                        )
);

INSTANTIATE_TEST_SUITE_P(conv_1x1s1p0_extra4, MyriadX_HW_Convolution_Tests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 512, 45, 80))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
                                , ::testing::Values<out_channels>(392)
                                , ::testing::Values<group>(1)
                                , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 1))
                        )
);

INSTANTIATE_TEST_SUITE_P(conv_1x1s2p0_extra1, MyriadX_HW_Convolution_Tests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 64, 180, 320))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 2, 2))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
                                , ::testing::Values<out_channels>(128)
                                , ::testing::Values<group>(1)
                                , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 1))
                        )
);

INSTANTIATE_TEST_SUITE_P(conv_1x1s2p0_extra2, MyriadX_HW_Convolution_Tests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 128, 90, 160))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 2, 2))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
                                , ::testing::Values<out_channels>(256)
                                , ::testing::Values<group>(1)
                                , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 1))
                        )
);

INSTANTIATE_TEST_SUITE_P(conv_1x1s2p0_extra3, MyriadX_HW_Convolution_Tests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 128, 90, 160))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 2, 2))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
                                , ::testing::Values<out_channels>(128)
                                , ::testing::Values<group>(1)
                                , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 1))
                        )
);

INSTANTIATE_TEST_SUITE_P(conv_3x3s2p1_extra1, MyriadX_HW_Convolution_Tests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 64, 180, 320))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 2, 2))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<out_channels>(128)
                                , ::testing::Values<group>(1)
                                , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 1))
                        )
);

INSTANTIATE_TEST_SUITE_P(conv_3x3s2p1_extra2, MyriadX_HW_Convolution_Tests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 128, 90, 160))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 2, 2))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<out_channels>(256)
                                , ::testing::Values<group>(1)
                                , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 1))
                        )
);

INSTANTIATE_TEST_SUITE_P(conv_3x3s2p1_extra3, MyriadX_HW_Convolution_Tests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 256, 45, 80))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 2, 2))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<out_channels>(512)
                                , ::testing::Values<group>(1)
                                , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 1))
                        )
);

INSTANTIATE_TEST_SUITE_P(conv_3x3s2p1_extra4, MyriadX_HW_Convolution_Tests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 32, 180, 320))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 2, 2))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<out_channels>(64)
                                , ::testing::Values<group>(1)
                                , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 1))
                        )
);

INSTANTIATE_TEST_SUITE_P(conv_1x1s1p0, MyriadX_HW_Convolution_Tests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 64, 56, 56))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
                                , ::testing::Values<out_channels>(64)
                                , ::testing::Values<group>(1)
                                , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 1))
                        )
);

INSTANTIATE_TEST_SUITE_P(conv_1x1s1p0_resnet50, MyriadX_HW_Convolution_Tests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 2048, 7, 7))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
                                , ::testing::Values<out_channels>(512)
                                , ::testing::Values<group>(1)
                                , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 1))
                        )
);

// This case adds extra CopyMakeBorder stage
INSTANTIATE_TEST_SUITE_P(conv_1x1s1p1, MyriadX_HW_Convolution_Tests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 512, 13, 13))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<out_channels>(1000)
                                , ::testing::Values<group>(1)
                                , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 1))
                        )
);

INSTANTIATE_TEST_SUITE_P(conv_1x1s2p0, MyriadX_HW_Convolution_Tests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 256, 56, 56),
                                                             MAKE_STRUCT(tensor_test_params, 1, 512, 28, 28),
                                                             MAKE_STRUCT(tensor_test_params, 1, 1024, 14, 14))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 2, 2))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
                                , ::testing::Values<out_channels>(128, 256, 512, 1024)
                                , ::testing::Values<group>(1)
                                , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 1))
                        )
);

INSTANTIATE_TEST_SUITE_P(conv_3x3s1p1, MyriadX_HW_Convolution_Tests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 64, 56, 56))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<out_channels>(192)
                                , ::testing::Values<group>(1), ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 1))

                        )
);

INSTANTIATE_TEST_SUITE_P(conv_3x3s1p1_yolo_tiny_v1_conv1, MyriadX_HW_Convolution_Tests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 3, 448, 448))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<out_channels>(16)
                                , ::testing::Values<group>(1)
                                , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 1))
                        )
);

INSTANTIATE_TEST_SUITE_P(conv_3x3s1p1_yolo_tiny_v1_conv7, MyriadX_HW_Convolution_Tests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 512, 7, 7))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<out_channels>(1024)
                                , ::testing::Values<group>(1)
                                , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 1))
                        )
);

INSTANTIATE_TEST_SUITE_P(conv_3x3s1p1_yolo_tiny_v1_conv8, MyriadX_HW_Convolution_Tests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 1024, 7, 7))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<out_channels>(256)
                                , ::testing::Values<group>(1)
                                , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 1))
                        )
);

INSTANTIATE_TEST_SUITE_P(conv_3x3s1p1_vgg, MyriadX_HW_Convolution_Tests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 64, 224, 224))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<out_channels>(64)
                                , ::testing::Values<group>(1)
                                , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 1))
                        )
);

INSTANTIATE_TEST_SUITE_P(conv_7x7s2p3, MyriadX_HW_Convolution_Tests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 3, 224, 224))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 7, 7))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 2, 2))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 3, 3))
                                , ::testing::Values<out_channels>(64)
                                , ::testing::Values<group>(1)
                                , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 1))
                        )
);

//  This case for unsymmetric convolution

INSTANTIATE_TEST_SUITE_P(conv_3x1s1_LPR, MyriadX_HW_Convolution_Tests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 32, 22, 92))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 1))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 0))
                                , ::testing::Values<out_channels>(32)
                                , ::testing::Values<group>(1)
                                , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 1))
                        )
);

INSTANTIATE_TEST_SUITE_P(conv_1x3s1_LPR, MyriadX_HW_Convolution_Tests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 32, 22, 92))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 1, 3))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 1))
                                , ::testing::Values<out_channels>(32)
                                , ::testing::Values<group>(1)
                                , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 1))
                        )
);

INSTANTIATE_TEST_SUITE_P(conv_1x5s1_LPR, MyriadX_HW_Convolution_Tests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 256, 5, 88))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 1, 5))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
                                , ::testing::Values<out_channels>(128)
                                , ::testing::Values<group>(1)
                                , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 1))
                        )
);

INSTANTIATE_TEST_SUITE_P(conv_13x1s1_LPR, MyriadX_HW_Convolution_Tests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 128, 1, 88))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 13, 1))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 6, 0))
                                , ::testing::Values<out_channels>(71)
                                , ::testing::Values<group>(1)
                                , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 1))
                        )
);

INSTANTIATE_TEST_SUITE_P(conv_5x1s1_LPR, MyriadX_HW_Convolution_Tests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 128, 1, 28))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 5, 1))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
                                , ::testing::Values<out_channels>(128)
                                , ::testing::Values<group>(1)
                                , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 1))
                        )
);

INSTANTIATE_TEST_SUITE_P(conv_4x4s2p1, MyriadX_HW_Convolution_Tests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 3, 256, 416))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 4, 4))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 2, 2))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<out_channels>(16)
                                , ::testing::Values<group>(1)
                                , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 1))
                        )
);

INSTANTIATE_TEST_SUITE_P(conv_5x5s2p1, MyriadX_HW_Convolution_Tests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 3, 256, 416))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 5, 5))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 2, 2))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<out_channels>(16)
                                , ::testing::Values<group>(1)
                                , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 1))
                        )
);

INSTANTIATE_TEST_SUITE_P(conv_5x5s2p2, MyriadX_HW_Convolution_Tests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 3, 256, 416))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 5, 5))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 2, 2))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 2, 2))
                                , ::testing::Values<out_channels>(16)
                                , ::testing::Values<group>(1)
                                , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 1))
                        )
);

INSTANTIATE_TEST_SUITE_P(conv_3x3s1p1_group1, MyriadX_HW_Convolution_Tests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 32, 150, 150))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<out_channels>(32)
                                , ::testing::Values<group>(32)
                                , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 1))
                        )
);

INSTANTIATE_TEST_SUITE_P(conv_3x3s2p1_group1, MyriadX_HW_Convolution_Tests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 64, 150, 150))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 2, 2))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<out_channels>(64)
                                , ::testing::Values<group>(1)
                                , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 1))
                        )
);

INSTANTIATE_TEST_SUITE_P(conv_3x3s2p1_group2, MyriadX_HW_Convolution_Tests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 128, 75, 75))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 2, 2))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<out_channels>(128)
                                , ::testing::Values<group>(128)
                                , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 1))
                        )
);

INSTANTIATE_TEST_SUITE_P(conv_3x3s2p1_group3, MyriadX_HW_Convolution_Tests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 256, 38, 38))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 2, 2))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<out_channels>(256)
                                , ::testing::Values<group>(256)
                                , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 1))
                        )
);

INSTANTIATE_TEST_SUITE_P(conv_3x3s1p1_pva_pvd, MyriadX_HW_Convolution_Tests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 6, 208, 368))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<out_channels>(16)
                                , ::testing::Values<group>(1)
                                , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 1))
                        )
);

INSTANTIATE_TEST_SUITE_P(conv_1x1s2p0_pva_pvd, MyriadX_HW_Convolution_Tests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 32, 128, 208))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 2, 2))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
                                , ::testing::Values<out_channels>(48)
                                , ::testing::Values<group>(1)
                                , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 1))
                        )
);

INSTANTIATE_TEST_SUITE_P(conv_3x3s1p1_ssd, MyriadX_HW_Convolution_Tests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 128, 75, 75))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<out_channels>(256)
                                , ::testing::Values<group>(1)
                                , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 1))
                        )
);

INSTANTIATE_TEST_SUITE_P(conv_unequal_hw_pad, MyriadX_HW_Convolution_Tests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 32, 128, 128))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 5, 5),
                                                            MAKE_STRUCT(param_size, 5, 1))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 2, 0))
                                , ::testing::Values<out_channels>(64)
                                , ::testing::Values<group>(1)
                                , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 1))
                        )
);

INSTANTIATE_TEST_SUITE_P(conv_3x3s3p1_resnet34, MyriadX_HW_Convolution_Tests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 512, 75, 75))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 3, 3))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<out_channels>(24)
                                , ::testing::Values<group>(1)
                                , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 1))
                        )
);

INSTANTIATE_TEST_SUITE_P(fc_to_conv_case, MyriadX_HW_Convolution_Tests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 256, 56, 350))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 7, 7))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 7, 7))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
                                , ::testing::Values<out_channels>(1024)
                                , ::testing::Values<group>(1)
                                , ::testing::Values<dilation_factor>(MAKE_STRUCT(param_size, 1, 1))
                        )
);

using HWConvPoolMerge = std::tuple<DimsInput,
                                   kernel, stride, pad, out_channels,
                                   kernel, stride, pad>;

class MyriadX_HW_ConvPoolMerged_Tests_nightly
        : public MyriadX_HW_Tests_nightly,
          public testing::WithParamInterface<HWConvPoolMerge> {
public:
    tensor_test_params in_dims;
    param_size conv_kernel;
    param_size conv_stride;
    param_size conv_pad;
    size_t conv_out_c;
    param_size pool_kernel;
    param_size pool_stride;
    param_size pool_pad;

    tensor_test_params conv_out_dims;

    size_t conv_num_weights;
    size_t conv_num_biases;

    tensor_test_params pool_out_dims;

    IN_OUT_desc in_tensor, conv_out_tensor, pool_out_tensor;

    void SetUp() override {
        ASSERT_NO_FATAL_FAILURE(MyriadX_HW_Tests_nightly::SetUp());

        in_dims = std::get<0>(GetParam());
        conv_kernel = std::get<1>(GetParam());
        conv_stride = std::get<2>(GetParam());
        conv_pad = std::get<3>(GetParam());
        conv_out_c = std::get<4>(GetParam());
        pool_kernel = std::get<5>(GetParam());
        pool_stride = std::get<6>(GetParam());
        pool_pad = std::get<7>(GetParam());

        size_t conv_out_w = (in_dims.w + 2 * conv_pad.x - conv_kernel.x + conv_stride.x) / conv_stride.x;
        size_t conv_out_h = (in_dims.h + 2 * conv_pad.y - conv_kernel.y + conv_stride.y) / conv_stride.y;
        conv_out_dims = {1, conv_out_c, conv_out_h, conv_out_w};

        conv_num_weights = conv_kernel.x * conv_kernel.y * in_dims.c * conv_out_dims.c;
        conv_num_biases = conv_out_dims.c;

        size_t pool_out_w = std::ceil((conv_out_dims.w + 2.0 * pool_pad.x - pool_kernel.x) / pool_stride.x + 1);
        size_t pool_out_h = std::ceil((conv_out_dims.h + 2.0 * pool_pad.y - pool_kernel.y) / pool_stride.y + 1);
        pool_out_dims = {1, conv_out_dims.c, pool_out_h, pool_out_w};

        in_tensor.push_back({in_dims.n, in_dims.c, in_dims.h, in_dims.w});
        conv_out_tensor.push_back({conv_out_dims.n, conv_out_dims.c, conv_out_dims.h, conv_out_dims.w});
        pool_out_tensor.push_back({pool_out_dims.n, pool_out_dims.c, pool_out_dims.h, pool_out_dims.w});
    }

    void AddConvLayer() {
        ParamsStruct conv_params = {
                  {"kernel-x", std::to_string(conv_kernel.x)}
                , {"kernel-y", std::to_string(conv_kernel.y)}
                , {"stride-x", std::to_string(conv_stride.x)}
                , {"stride-y", std::to_string(conv_stride.y)}
                , {"pad-x", std::to_string(conv_pad.x)}
                , {"pad-y", std::to_string(conv_pad.y)}
                , {"output", std::to_string(conv_out_dims.c)}
                , {"group", "1"}
        };
        _testNet.addLayer(LayerInitParams("Convolution")
                 .params(conv_params)
                 .weights(conv_num_weights).fillWeights(defaultWeightsRange)
                 .biases(conv_num_biases).fillBiases(defaultWeightsRange)
                 .in(in_tensor)
                 .out(conv_out_tensor),
                 ref_convolution_wrap);
    }

    void AddReLULayer(float negativeSlope) {
        ParamsStruct relu_params = {
            {"negative_slope", std::to_string(negativeSlope)}
        };
        _testNet.addLayer(LayerInitParams("ReLU")
                 .params(relu_params)
                 .in(conv_out_tensor)
                 .out(conv_out_tensor),
                 ref_ReLU_wrap);
    }

    void AddPoolLayer() {
        ParamsStruct pool_params = {
                  {"kernel-x", std::to_string(pool_kernel.x)}
                , {"kernel-y", std::to_string(pool_kernel.y)}
                , {"stride-x", std::to_string(pool_stride.x)}
                , {"stride-y", std::to_string(pool_stride.y)}
                , {"pad-x", std::to_string(pool_pad.x)}
                , {"pad-y", std::to_string(pool_pad.y)}
                , {"pool-method", "max"}
        };
        _testNet.addLayer(LayerInitParams("Pooling")
                 .params(pool_params)
                 .in(conv_out_tensor)
                 .out(pool_out_tensor),
                 ref_pooling_wrap);
    }
};

TEST_P(MyriadX_HW_ConvPoolMerged_Tests_nightly, WithReLU) {
    if (!CheckMyriadX()) {
        GTEST_SKIP() << "Non-MyriadX device";
    }

    AddConvLayer();
    AddReLULayer(0.0f);
    AddPoolLayer();

    auto maxerr = 0.0009 * in_dims.c * conv_kernel.x * conv_kernel.y;
    CompareWithSW(maxerr);
}

TEST_P(MyriadX_HW_ConvPoolMerged_Tests_nightly, WithLeakyReLU) {
    if (!CheckMyriadX()) {
        GTEST_SKIP() << "Non-MyriadX device";
    }

    AddConvLayer();
    AddReLULayer(0.1f);
    AddPoolLayer();

    auto maxerr = 0.01 * in_dims.c * conv_kernel.x * conv_kernel.y;
    CompareWithSW(maxerr);
}

INSTANTIATE_TEST_SUITE_P(yolo_conv1, MyriadX_HW_ConvPoolMerged_Tests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 3, 448, 448)),
                                ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3)),
                                ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1)),
                                ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1)),
                                ::testing::Values<out_channels>(16),
                                ::testing::Values<kernel>(MAKE_STRUCT(param_size, 2, 2)),
                                ::testing::Values<stride>(MAKE_STRUCT(param_size, 2, 2)),
                                ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
                        )
);

INSTANTIATE_TEST_SUITE_P(yolov2_tf_conv, MyriadX_HW_ConvPoolMerged_Tests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 32, 304, 304),
                                                             MAKE_STRUCT(tensor_test_params, 1, 64, 152, 152)),
                                ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3)),
                                ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1)),
                                ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1)),
                                ::testing::Values<out_channels>(64, 128),
                                ::testing::Values<kernel>(MAKE_STRUCT(param_size, 2, 2)),
                                ::testing::Values<stride>(MAKE_STRUCT(param_size, 2, 2)),
                                ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
                        )
);

INSTANTIATE_TEST_SUITE_P(yolo_conv2, MyriadX_HW_ConvPoolMerged_Tests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 16, 224, 224)),
                                ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3)),
                                ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1)),
                                ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1)),
                                ::testing::Values<out_channels>(32),
                                ::testing::Values<kernel>(MAKE_STRUCT(param_size, 2, 2)),
                                ::testing::Values<stride>(MAKE_STRUCT(param_size, 2, 2)),
                                ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
                        )
);

INSTANTIATE_TEST_SUITE_P(yolo_conv4, MyriadX_HW_ConvPoolMerged_Tests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 64, 56, 56)),
                                ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3)),
                                ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1)),
                                ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1)),
                                ::testing::Values<out_channels>(128),
                                ::testing::Values<kernel>(MAKE_STRUCT(param_size, 2, 2)),
                                ::testing::Values<stride>(MAKE_STRUCT(param_size, 2, 2)),
                                ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
                        )
);

INSTANTIATE_TEST_SUITE_P(ssd_case1, MyriadX_HW_ConvPoolMerged_Tests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 64, 98, 150)),
                                ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3)),
                                ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1)),
                                ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1)),
                                ::testing::Values<out_channels>(64),
                                ::testing::Values<kernel>(MAKE_STRUCT(param_size, 2, 2)),
                                ::testing::Values<stride>(MAKE_STRUCT(param_size, 2, 2)),
                                ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
                        )
);

INSTANTIATE_TEST_SUITE_P(vgg16_case1, MyriadX_HW_ConvPoolMerged_Tests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 512, 28, 28)),
                                ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3)),
                                ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1)),
                                ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1)),
                                ::testing::Values<out_channels>(512),
                                ::testing::Values<kernel>(MAKE_STRUCT(param_size, 2, 2)),
                                ::testing::Values<stride>(MAKE_STRUCT(param_size, 2, 2)),
                                ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
                        )
);

using ConvTFParams = std::tuple<DimsInput, DimsOutput, kernel, stride, tfPad, group>;

class MyriadX_HW_ConvTF_Tests_nightly :
        public MyriadX_HW_Tests_nightly,
        public testing::WithParamInterface<ConvTFParams>{
public:
    tensor_test_params inDims;
    tensor_test_params outDims;
    param_size kernel;
    param_size stride;
    paddings4 pad;
    int group;

    void SetUp() override {
        ASSERT_NO_FATAL_FAILURE(MyriadX_HW_Tests_nightly::SetUp());

        inDims = std::get<0>(GetParam());
        outDims = std::get<1>(GetParam());
        kernel = std::get<2>(GetParam());
        stride = std::get<3>(GetParam());
        pad = std::get<4>(GetParam());
        group = std::get<5>(GetParam());
    }

    void AddConvolutionLayer() {
        std::map<std::string, std::string> convParams = {
            {"kernel-x", std::to_string(kernel.x)},
            {"kernel-y", std::to_string(kernel.y)},
            {"stride-x", std::to_string(stride.x)},
            {"stride-y", std::to_string(stride.y)},
            {"pad-x", std::to_string(pad.left)},
            {"pad-r", std::to_string(pad.right)},
            {"pad-y", std::to_string(pad.top)},
            {"pad-b", std::to_string(pad.bottom)},
            {"dilation-x", "1"},
            {"dilation-y", "1"},
            {"group", std::to_string(group)},
            {"output", std::to_string(outDims.c)}
        };

        _testNet.addLayer(LayerInitParams("Convolution")
                 .params(convParams)
                 .weights(kernel.x * kernel.y * (inDims.c / group) * outDims.c)
                 .biases(outDims.c)
                 .fillWeights(defaultWeightsRange)
                 .fillBiases(defaultWeightsRange)
                 .in({{inDims.n, inDims.c, inDims.h, inDims.w}})
                 .out({{outDims.n, outDims.c, outDims.h, outDims.w}}),
            ref_convolution_wrap);
    }
};

TEST_P(MyriadX_HW_ConvTF_Tests_nightly, Single) {
    if (!CheckMyriadX()) {
        GTEST_SKIP() << "Non-MyriadX device";
    }

    AddConvolutionLayer();

    float maxerr = 0.002 * (inDims.c / group) * kernel.x * kernel.y;
    CompareWithSW(maxerr);
}

INSTANTIATE_TEST_SUITE_P(tf, MyriadX_HW_ConvTF_Tests_nightly,
    ::testing::Values(
        std::make_tuple(
            MAKE_STRUCT(tensor_test_params, 1, 3, 224, 224),    // input
            MAKE_STRUCT(tensor_test_params, 1, 24, 112, 112),   // output
            MAKE_STRUCT(param_size, 7, 7),                      // kernel
            MAKE_STRUCT(param_size, 2, 2),                      // stride
            MAKE_STRUCT(paddings4, 2, 2, 3, 3),                 // pad
            3                                                   // group
        ),
        std::make_tuple(
            MAKE_STRUCT(tensor_test_params, 1, 64, 56, 56),     // input
            MAKE_STRUCT(tensor_test_params, 1, 192, 56, 56),    // output
            MAKE_STRUCT(param_size, 3, 3),                      // kernel
            MAKE_STRUCT(param_size, 1, 1),                      // stride
            MAKE_STRUCT(paddings4, 1, 1, 1, 1),                 // pad
            1                                                   // group
        ),
        std::make_tuple(
            MAKE_STRUCT(tensor_test_params, 1, 128, 3, 3),      // input
            MAKE_STRUCT(tensor_test_params, 1, 128, 2, 2),      // output
            MAKE_STRUCT(param_size, 3, 3),                      // kernel
            MAKE_STRUCT(param_size, 2, 2),                      // stride
            MAKE_STRUCT(paddings4, 1, 1, 1, 1),                 // pad
            128                                                 // group
        ),
        std::make_tuple(
            MAKE_STRUCT(tensor_test_params, 1, 256, 2, 2),      // input
            MAKE_STRUCT(tensor_test_params, 1, 24, 2, 2),       // output
            MAKE_STRUCT(param_size, 3, 3),                      // kernel
            MAKE_STRUCT(param_size, 1, 1),                      // stride
            MAKE_STRUCT(paddings4, 1, 1, 1, 1),                 // pad
            1                                                   // group
        ),
        std::make_tuple(
            MAKE_STRUCT(tensor_test_params, 1, 64, 2, 2),       // input
            MAKE_STRUCT(tensor_test_params, 1, 64, 1, 1),       // output
            MAKE_STRUCT(param_size, 3, 3),                      // kernel
            MAKE_STRUCT(param_size, 2, 2),                      // stride
            MAKE_STRUCT(paddings4, 0, 0, 1, 1),                 // pad
            64                                                  // group
        ),
        std::make_tuple(
            MAKE_STRUCT(tensor_test_params, 1, 128, 1, 1),      // input
            MAKE_STRUCT(tensor_test_params, 1, 24, 1, 1),       // output
            MAKE_STRUCT(param_size, 3, 3),                      // kernel
            MAKE_STRUCT(param_size, 1, 1),                      // stride
            MAKE_STRUCT(paddings4, 1, 1, 1, 1),                 // pad
            1                                                   // group
        ),
        std::make_tuple(
            MAKE_STRUCT(tensor_test_params, 1, 32, 128, 128),   // input
            MAKE_STRUCT(tensor_test_params, 1, 64, 128, 128),   // output
            MAKE_STRUCT(param_size, 2, 2),                      // kernel
            MAKE_STRUCT(param_size, 1, 1),                      // stride
            MAKE_STRUCT(paddings4, 0, 0, 1, 1),                 // pad
            1                                                   // group
        ),
        std::make_tuple(
            MAKE_STRUCT(tensor_test_params, 1, 32, 128, 128),   // input
            MAKE_STRUCT(tensor_test_params, 1, 64, 128, 128),   // output
            MAKE_STRUCT(param_size, 2, 2),                      // kernel
            MAKE_STRUCT(param_size, 1, 1),                      // stride
            MAKE_STRUCT(paddings4, 1, 1, 0, 0),                 // pad
            1                                                   // group
        )
    )
);

using HWDeconvParams = std::tuple<DimsInput, kernel, stride, pad, out_channels, group>;

class MyriadX_HW_Deconvolution_Tests_nightly
        : public MyriadX_HW_Tests_nightly,
          public testing::WithParamInterface<HWDeconvParams> {
public:
    tensor_test_params in_dims;
    param_size kernel;
    param_size stride;
    param_size pad;
    size_t out_c;
    size_t group;

    tensor_test_params out_dims;

    IN_OUT_desc in_tensor, out_tensor;

    size_t numWeights;
    size_t numBiases;

    void SetUp() override {
        ASSERT_NO_FATAL_FAILURE(MyriadX_HW_Tests_nightly::SetUp());

        in_dims = std::get<0>(GetParam());
        kernel = std::get<1>(GetParam());
        stride = std::get<2>(GetParam());
        pad = std::get<3>(GetParam());
        out_c = std::get<4>(GetParam());
        group = std::get<5>(GetParam());

        size_t out_w = stride.x * (in_dims.w - 1) + kernel.x - 2 * pad.x;
        size_t out_h = stride.y * (in_dims.h - 1) + kernel.y - 2 * pad.y;
        out_dims = {1, out_c, out_h, out_w};

        in_tensor.push_back({in_dims.n, in_dims.c, in_dims.h, in_dims.w});
        out_tensor.push_back({out_dims.n, out_dims.c, out_dims.h, out_dims.w});

        numWeights = kernel.x * kernel.y * (in_dims.c / group) * out_dims.c;
        numBiases = out_dims.c;
    }

    void AddInitialCopyLayer() {
        _testNet.addLayer(LayerInitParams("Copy")
                 .in(in_tensor)
                 .out(in_tensor),
                 ref_copy_wrap);
    }

    void AddDeconvolutionLayer() {
        std::map<std::string, std::string> deconvParams = {
                  {"kernel-x", std::to_string(kernel.x)}
                , {"kernel-y", std::to_string(kernel.y)}
                , {"stride-x", std::to_string(stride.x)}
                , {"stride-y", std::to_string(stride.y)}
                , {"pad-x", std::to_string(pad.x)}
                , {"pad-y", std::to_string(pad.y)}
                , {"output", std::to_string(out_c)}
                , {"group", std::to_string(group)}
        };
        _testNet.addLayer(LayerInitParams("Deconvolution")
                 .params(deconvParams)
                 .weights(numWeights)
                 .biases(numBiases)
                 .fillWeights(defaultWeightsRange)
                 .fillBiases(defaultWeightsRange)
                 .in(in_tensor)
                 .out(out_tensor),
                 ref_deconvolution_wrap);
    }

    void AddDeconvolutionLayerSmallWeights() {
        std::map<std::string, std::string> deconvParams = {
                  {"kernel-x", std::to_string(kernel.x)}
                , {"kernel-y", std::to_string(kernel.y)}
                , {"stride-x", std::to_string(stride.x)}
                , {"stride-y", std::to_string(stride.y)}
                , {"pad-x", std::to_string(pad.x)}
                , {"pad-y", std::to_string(pad.y)}
                , {"output", std::to_string(out_c)}
                , {"group", std::to_string(group)}
        };
        _testNet.addLayer(LayerInitParams("Deconvolution")
                 .params(deconvParams)
                 .weights(numWeights)
                 .biases(numBiases)
                 .fillWeights(smallWeightsRange)
                 .fillBiases(smallWeightsRange)
                 .in(in_tensor)
                 .out(out_tensor),
                 ref_deconvolution_wrap);
    }
};

TEST_P(MyriadX_HW_Deconvolution_Tests_nightly, Single) {
    if (!CheckMyriadX()) {
        GTEST_SKIP() << "Non-MyriadX device";
    }

    AddInitialCopyLayer();
    AddDeconvolutionLayer();

    float maxerr = 0.002 * (in_dims.c / group) * kernel.x * kernel.y;
    CompareWithSW(maxerr);
}

TEST_P(MyriadX_HW_Deconvolution_Tests_nightly, ScaleTests) {
    if (!CheckMyriadX()) {
        GTEST_SKIP() << "Non-MyriadX device";
    }

    AddInitialCopyLayer();
    AddDeconvolutionLayerSmallWeights();

    float maxerr = 0.01;
    CompareWithSW(maxerr);
}

INSTANTIATE_TEST_SUITE_P(deconv_tf_ssd, MyriadX_HW_Deconvolution_Tests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 2, 3, 3))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
                                , ::testing::Values<out_channels>(1)
                                , ::testing::Values<group>(1)
                        )
);

INSTANTIATE_TEST_SUITE_P(deconv_3x3_str1, MyriadX_HW_Deconvolution_Tests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 256, 3, 3),
                                                             MAKE_STRUCT(tensor_test_params, 1, 128, 5, 5),
                                                             MAKE_STRUCT(tensor_test_params, 1, 64, 7, 7))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
                                , ::testing::Values<out_channels>(128, 256)
                                , ::testing::Values<group>(1)
                        )
);

INSTANTIATE_TEST_SUITE_P(hw_accuracy_deconv_3x3, MyriadX_HW_Deconvolution_Tests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 256, 5, 5),
                                                             MAKE_STRUCT(tensor_test_params, 1, 128, 11, 11),
                                                             MAKE_STRUCT(tensor_test_params, 1, 64, 13, 13),
                                                             MAKE_STRUCT(tensor_test_params, 1, 32, 8, 8))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 2, 2))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0),
                                                         MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<out_channels>(1, 128)
                                , ::testing::Values<group>(1)
                        )
);

INSTANTIATE_TEST_SUITE_P(hw_accuracy_deconv, MyriadX_HW_Deconvolution_Tests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 120, 36, 36),
                                                             MAKE_STRUCT(tensor_test_params, 1, 73, 40, 54),
                                                             MAKE_STRUCT(tensor_test_params, 1, 7, 9, 13))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 5, 5))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1),
                                                            MAKE_STRUCT(param_size, 2, 2))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0),
                                                         MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<out_channels>(19, 53)
                                , ::testing::Values<group>(1)
                        )
);

INSTANTIATE_TEST_SUITE_P(hw_accuracy_scale_deconv, MyriadX_HW_Deconvolution_Tests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 120, 36, 36))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 5, 5))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1),
                                                            MAKE_STRUCT(param_size, 2, 2))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0),
                                                         MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<out_channels>(256)
                                , ::testing::Values<group>(1)
                        )
);
