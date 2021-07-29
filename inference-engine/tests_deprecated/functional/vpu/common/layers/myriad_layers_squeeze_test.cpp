// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_squeeze_test.hpp"

INSTANTIATE_TEST_SUITE_P(accuracy, myriadLayersTestsSqueezeTC1_smoke,
    ::testing::Combine(
        ::testing::ValuesIn(s_squeezeTensorsTC1),
        ::testing::ValuesIn(s_squeezeIndicesTC1),
        ::testing::ValuesIn(s_squeezeKeepAtLeast1D),
        ::testing::ValuesIn(s_squeezeLayoutPreference)
    )
);

INSTANTIATE_TEST_SUITE_P(accuracy, myriadLayersTestsSqueezeTC2_smoke,
    ::testing::Combine(
        ::testing::ValuesIn(s_squeezeTensorsTC2),
        ::testing::ValuesIn(s_squeezeIndicesTC2),
        ::testing::ValuesIn(s_squeezeKeepAtLeast1D),
        ::testing::ValuesIn(s_squeezeLayoutPreference)
    )
);

// TODO: rewrite to ngraph to have reshape functionality
INSTANTIATE_TEST_SUITE_P(DISABLED_accuracy, myriadLayersTestsSqueezeTC3_smoke,
    ::testing::Combine(
        ::testing::ValuesIn(s_squeezeTensorsTC3),
        ::testing::ValuesIn(s_squeezeIndicesTC3),
        ::testing::ValuesIn(s_squeezeKeepAtLeast1D),
        ::testing::ValuesIn(s_squeezeLayoutPreference)
    )
);

// TODO: rewrite to ngraph to have reshape functionality
INSTANTIATE_TEST_SUITE_P(DISABLED_accuracy, myriadLayersTestsSqueezeTC4_smoke,
        ::testing::Combine(
        ::testing::ValuesIn(s_squeezeTensorsTC4),
        ::testing::ValuesIn(s_squeezeIndicesTC4),
        ::testing::ValuesIn(s_squeezeKeepAtLeast1D),
        ::testing::ValuesIn(s_squeezeLayoutPreference)
    )
);

INSTANTIATE_TEST_SUITE_P(accuracy, myriadLayersTestsSqueezeTC5_smoke,
                        ::testing::Combine(
                                ::testing::ValuesIn(s_squeezeTensorsTC5),
                                ::testing::ValuesIn(s_squeezeIndicesTC5),
                                ::testing::ValuesIn(s_squeezeKeepAtLeast1D),
                                ::testing::ValuesIn(s_squeezeLayoutPreference)
                        )
);