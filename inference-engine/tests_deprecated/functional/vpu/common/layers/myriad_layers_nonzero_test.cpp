// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_nonzero_test.hpp"

INSTANTIATE_TEST_CASE_P(accuracy, myriadLayerTestNonZero_smoke,
                        ::testing::ValuesIn(inputDims));
