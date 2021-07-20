// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_softmax_test.hpp"

INSTANTIATE_TEST_SUITE_P(
    accuracy, myriadLayersTestsSoftMax_smoke,    
    ::testing::Combine(
        ::testing::ValuesIn(s_softMaxTensors)
      , ::testing::Values<IRVersion>(IRVersion::v7, IRVersion::v10)
      )
);
