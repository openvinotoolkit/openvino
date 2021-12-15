// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_tile_test.hpp"

INSTANTIATE_TEST_SUITE_P(accuracyAdd, myriadLayerTestTile_smoke,
                        ::testing::Combine(
                                ::testing::Values<test_params>(
                                        MAKE_STRUCT(tile_test::nd_tensor_test_params, {4, 5, 6}, 0)
                                        , MAKE_STRUCT(tile_test::nd_tensor_test_params, {4, 5, 6, 7}, 0)
                                        , MAKE_STRUCT(tile_test::nd_tensor_test_params, {4, 5, 6, 7}, 1)
                                        , MAKE_STRUCT(tile_test::nd_tensor_test_params, {4, 5, 6, 27, 13}, 0)
                                        , MAKE_STRUCT(tile_test::nd_tensor_test_params, {4, 5, 6, 27, 13}, 1)
                                        , MAKE_STRUCT(tile_test::nd_tensor_test_params, {4, 5, 6, 27, 13}, 2)
                                        , MAKE_STRUCT(tile_test::nd_tensor_test_params, {4, 5, 6, 27, 13, 18}, 0)
                                        , MAKE_STRUCT(tile_test::nd_tensor_test_params, {4, 5, 6, 27, 13, 18}, 1)
                                        , MAKE_STRUCT(tile_test::nd_tensor_test_params, {4, 5, 6, 27, 13, 18}, 2)
                                        , MAKE_STRUCT(tile_test::nd_tensor_test_params, {4, 5, 6, 27, 13, 18}, 3))

                                , ::testing::Values<tiles>(2, 3, 5)
                        ));

INSTANTIATE_TEST_SUITE_P(accuracy, myriadLayerTestTile_smoke,
                        ::testing::Combine(
                                ::testing::Values<test_params>(
                                        MAKE_STRUCT(tile_test::nd_tensor_test_params, {4, 5, 6}, 1)
                                        , MAKE_STRUCT(tile_test::nd_tensor_test_params, {4, 5, 6}, 2)
                                        , MAKE_STRUCT(tile_test::nd_tensor_test_params, {5, 6}, 0)
                                        , MAKE_STRUCT(tile_test::nd_tensor_test_params, {5, 6}, 1)
                                        , MAKE_STRUCT(tile_test::nd_tensor_test_params, {6}, 0)
                                        , MAKE_STRUCT(tile_test::nd_tensor_test_params, {6, 5, 6, 7}, 2)
                                        , MAKE_STRUCT(tile_test::nd_tensor_test_params, {6, 5, 6, 7}, 3)
                                        , MAKE_STRUCT(tile_test::nd_tensor_test_params, {4, 5, 6, 27, 13}, 3)
                                        , MAKE_STRUCT(tile_test::nd_tensor_test_params, {4, 5, 6, 27, 13}, 4)
                                        , MAKE_STRUCT(tile_test::nd_tensor_test_params, {4, 5, 6, 27, 13, 18}, 4)
                                        , MAKE_STRUCT(tile_test::nd_tensor_test_params, {4, 5, 6, 27, 13, 18}, 5))

                                , ::testing::Values<tiles>(2, 3, 5)
                        ));
