// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_unsqueeze_test.hpp"

INSTANTIATE_TEST_SUITE_P(accuracy, myriadLayersTestsUnsqueeze_smoke,
    ::testing::Combine(
        ::testing::ValuesIn(s_squeezeTensors),
        ::testing::ValuesIn(s_squeezeIndices)
    )
);