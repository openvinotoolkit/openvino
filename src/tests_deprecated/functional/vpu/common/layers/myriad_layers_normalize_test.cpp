// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_normalize_test.hpp"

INSTANTIATE_TEST_SUITE_P(accuracy, myriadLayersTestsNormalize_smoke, ::testing::Combine(
    ::testing::Values<Dims>(
        // small size, num_channels is not divisible by 8
        MAKE_STRUCT(tensor_test_params, 1, 33, 1, 1),

        // size used in SSD_VGG topology
        MAKE_STRUCT(tensor_test_params, 1, 512, 38, 38),

        // size used in a customer topology
        MAKE_STRUCT(tensor_test_params, 1, 128, 1, 1)
    ),
    ::testing::Values<AcrossSpatial>(false, true),
    ::testing::Values<ChannelSharedNormalize>(false, true),
    ::testing::Values<EPS>(1e-10f, 1e-9f, 1e-8f, 1e-7f, 1.192093e-07, 1e-6f, 1e-5f, 1e-4f, 1e-3f, 0.0f)
));


INSTANTIATE_TEST_SUITE_P(accuracy_more, myriadLayersTestsNormalize_smoke, ::testing::Combine(
    ::testing::Values<Dims>(
        //more tests
        MAKE_STRUCT(tensor_test_params, 1, 1, 38, 38),
        MAKE_STRUCT(tensor_test_params, 1, 1, 1, 1),
        MAKE_STRUCT(tensor_test_params, 1, 1, 8, 8),
        MAKE_STRUCT(tensor_test_params, 1, 3, 17, 17),
        MAKE_STRUCT(tensor_test_params, 1, 1, 17, 17),
        MAKE_STRUCT(tensor_test_params, 1, 1, 32, 32),
        MAKE_STRUCT(tensor_test_params, 1, 8, 38, 38),
        MAKE_STRUCT(tensor_test_params, 1, 512, 1, 1),
        MAKE_STRUCT(tensor_test_params, 1, 512, 8, 8)
    ),
    ::testing::Values<AcrossSpatial>(false, true),
    ::testing::Values<ChannelSharedNormalize>(false, true),
    ::testing::Values<EPS>(1e-10f, 1e-9f, 1e-8f, 1e-7f, 1.192093e-07, 1e-6f, 1e-5f, 1e-4f, 1e-3f, 0.0f)
));
