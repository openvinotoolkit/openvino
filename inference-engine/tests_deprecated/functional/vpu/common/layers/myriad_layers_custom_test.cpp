// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_custom_test.hpp"

INSTANTIATE_TEST_CASE_P(accuracy, myriadLayersTestsShuffleChannel_nightly,
        ::testing::Combine(
        ::testing::ValuesIn(s_ShuffleChannelTensors),
        ::testing::ValuesIn(s_ShuffleChannelGroup),
        ::testing::ValuesIn(s_CustomConfig)));

INSTANTIATE_TEST_CASE_P(accuracy, myriadLayersTestsQuantize_nightly,
        ::testing::Combine(
        ::testing::ValuesIn(s_QuantizeTensors),
        ::testing::ValuesIn(s_QuantizeLevels),
        ::testing::ValuesIn(s_CustomConfig)));

INSTANTIATE_TEST_CASE_P(accuracy, myriadLayersTestsQuantizeBinarize_nightly,
        ::testing::Combine(
        ::testing::ValuesIn(s_QuantizeTensors),
        ::testing::ValuesIn(s_QuantizeLevels),
        ::testing::ValuesIn(s_QuantizeSwitchOut),
        ::testing::ValuesIn(s_CustomConfig)));

INSTANTIATE_TEST_CASE_P(accuracy, myriadLayersTestsBinaryConvolution_nightly,
        ::testing::Combine(
        ::testing::ValuesIn(s_BinaryConvolutionTensors),
        ::testing::ValuesIn(s_BinaryConvolutionDilations),
        ::testing::ValuesIn(s_BinaryConvolutionGroup),
        ::testing::ValuesIn(s_BinaryConvolutionKernel),
        ::testing::ValuesIn(s_BinaryConvolutionStrides),
        ::testing::ValuesIn(s_CustomConfig)));

INSTANTIATE_TEST_CASE_P(accuracy, myriadLayersTestsExperimentalDetectronPriorGridGenerator_nightly,
        ::testing::Combine(
        ::testing::ValuesIn(s_ExperimentalDetectronPriorGridGeneratorImageDims),
        ::testing::ValuesIn(s_CustomConfig)));
