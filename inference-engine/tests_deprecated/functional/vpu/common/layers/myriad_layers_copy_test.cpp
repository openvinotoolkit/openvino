// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_copy_test.hpp"

INSTANTIATE_TEST_SUITE_P(accuracy, myriadLayerCopy_smoke,
        ::testing::Combine(
            ::testing::Values<NDims>(MAKE_STRUCT(nd_tensor_test_params, {36, 19, 20, 21})
                                   , MAKE_STRUCT(nd_tensor_test_params, {7, 8, 5, 12})
                                   , MAKE_STRUCT(nd_tensor_test_params, {196, 12, 20, 5}))
          , ::testing::Values<int>(2, 3, 4)
                        ));
