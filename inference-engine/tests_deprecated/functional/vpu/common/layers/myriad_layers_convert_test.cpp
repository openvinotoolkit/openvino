// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_convert_test.hpp"

INSTANTIATE_TEST_CASE_P(
    accuracy, myriadLayersTestsIOConvert_nightly,
    ::testing::Combine(
        ::testing::ValuesIn(inputsDims),
        ::testing::ValuesIn(precisionsIO)
    )
);

INSTANTIATE_TEST_CASE_P(
    accuracy, myriadLayersTestsConvertWithFP16_nightly,
    ::testing::Combine(
        ::testing::ValuesIn(inputsDims),
        ::testing::ValuesIn(withFP16Precisions)
    )
);
