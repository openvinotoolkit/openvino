// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_custom_test.hpp"

INSTANTIATE_TEST_SUITE_P(accuracy, myriadLayersTestsShuffleChannel_smoke,
        ::testing::Combine(
        ::testing::ValuesIn(s_ShuffleChannelTensors),
        ::testing::ValuesIn(s_ShuffleChannelGroup),
        ::testing::ValuesIn(s_CustomConfig)));

INSTANTIATE_TEST_SUITE_P(accuracy, myriadLayersTestsFakeQuantize_smoke,
        ::testing::Combine(
        ::testing::ValuesIn(s_QuantizeTensors),
        ::testing::ValuesIn(s_QuantizeLevels),
        ::testing::Values(IRVersion::v7, IRVersion::v10),
        ::testing::ValuesIn(s_CustomConfig)));

INSTANTIATE_TEST_SUITE_P(accuracy, myriadLayersTestsQuantizeBinarize_smoke,
        ::testing::Combine(
        ::testing::ValuesIn(s_QuantizeTensors),
        ::testing::Values(2),
        ::testing::ValuesIn(s_QuantizeSwitchOut),
        ::testing::ValuesIn(s_CustomConfig)));

INSTANTIATE_TEST_SUITE_P(accuracy, myriadLayersTestsBinaryConvolution_smoke,
        ::testing::Combine(
        ::testing::ValuesIn(s_BinaryConvolutionTensors),
        ::testing::ValuesIn(s_BinaryConvolutionDilations),
        ::testing::ValuesIn(s_BinaryConvolutionGroup),
        ::testing::ValuesIn(s_BinaryConvolutionKernel),
        ::testing::ValuesIn(s_BinaryConvolutionStrides),
        ::testing::ValuesIn(s_CustomConfig)));

INSTANTIATE_TEST_SUITE_P(accuracy, myriadLayersTestsExperimentalDetectronPriorGridGenerator_smoke,
        ::testing::Combine(
        ::testing::ValuesIn(s_ExperimentalDetectronPriorGridGeneratorImageDims),
        ::testing::ValuesIn(s_CustomConfig)));

INSTANTIATE_TEST_SUITE_P(accuracy, myriadLayersTestsCorrelate_smoke,
        ::testing::Combine(
        ::testing::ValuesIn(s_CorrelateParams),
        ::testing::ValuesIn(s_CustomConfig)));

INSTANTIATE_TEST_SUITE_P(accuracy, myriadLayersTestsSpatialTransform_smoke,
        ::testing::Combine(
        ::testing::ValuesIn(s_SpatialTransformInputs),
        ::testing::ValuesIn(s_SpatialTransformTheta),
        ::testing::ValuesIn(s_CustomConfig)));
