// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_convert_test.hpp"

INSTANTIATE_TEST_SUITE_P(
    accuracy, myriadLayersTestsIOConvert_smoke,
    ::testing::Combine(
        ::testing::ValuesIn(inputsDims),
        ::testing::ValuesIn(precisionsIO),
        ::testing::Values("")
    )
);

INSTANTIATE_TEST_SUITE_P(
        accuracy_customu8f16, myriadLayersTestsIOConvert_smoke,
        ::testing::Combine(
                ::testing::ValuesIn(inputsDims4D),
                ::testing::Values(PrecisionPair{Precision::U8, Precision::FP16}),
                ::testing::Values(s_CustomConfig)
        )
);

INSTANTIATE_TEST_SUITE_P(
    accuracy, myriadLayersTestsConvertWithFP16_smoke,
    ::testing::Combine(
        ::testing::ValuesIn(inputsDims),
        ::testing::ValuesIn(withFP16Precisions)
    )
);
