// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_batch_normalization_test.hpp"

INSTANTIATE_TEST_SUITE_P(
        accuracy, myriadLayersTestsBatchNormalization_smoke,
        ::testing::Values(
                bn_test_params{{1, 1, 16, 8}, 0.001f},
                bn_test_params{{1, 4, 8, 16}, 0.00001f},
                bn_test_params{{1, 44, 88, 16}, 0.003f},
                bn_test_params{{1, 16, 32, 32}, 0.00005f},
                bn_test_params{{1, 512, 7, 7}, 0.0000096f}));
