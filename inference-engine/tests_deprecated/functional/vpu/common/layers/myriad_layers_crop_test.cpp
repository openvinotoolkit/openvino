// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_crop_test.hpp"

INSTANTIATE_TEST_SUITE_P(
    accuracy_Crop, myriadLayerCropOneInputAndDim_smoke,
    ::testing::Combine(
        ::testing::ValuesIn(s_tileTensors1),
        ::testing::ValuesIn(s_tileTensors2),
        ::testing::ValuesIn(s_tileCropAxis),
        ::testing::ValuesIn(s_tileOffset),
        ::testing::ValuesIn(s_tileDim))
);

INSTANTIATE_TEST_SUITE_P(
    accuracy_Crop1, myriadLayerCropOneInput_smoke,
    ::testing::Combine(
        ::testing::ValuesIn(s_tileTensors1),
        ::testing::ValuesIn(s_tileTensors2),
        ::testing::ValuesIn(s_tileCropAxis),
        ::testing::ValuesIn(s_tileCropBegin),
        ::testing::ValuesIn(s_tileCropEnd))
);

INSTANTIATE_TEST_SUITE_P(
    accuracy_Crop2, myriadLayerCropTwoInputs_smoke,
    ::testing::Combine(
        ::testing::ValuesIn(s_tileTensors1),
        ::testing::ValuesIn(s_tileTensors2),
        ::testing::ValuesIn(s_tileTensors2),
        ::testing::ValuesIn(s_tileCropAxis),
        ::testing::ValuesIn(s_tileOffset))
);