// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_reverse_sequence_test.hpp"

INSTANTIATE_TEST_SUITE_P(accuracy, myriadLayerReverseSequence_smoke,
    ::testing::Combine(
        ::testing::Values<ReverseSequence>(
                MAKE_STRUCT(reverse_sequence_test_params, {5, 6, 18}, 0, 0)
              , MAKE_STRUCT(reverse_sequence_test_params, {1,  2, 5, 2, 5}, 3, 4)
              , MAKE_STRUCT(reverse_sequence_test_params, {5, 6, 18}, 0, 1)
              , MAKE_STRUCT(reverse_sequence_test_params, {5, 6, 18}, 0, 2)
              , MAKE_STRUCT(reverse_sequence_test_params, {1,  4, 2, 5}, 2, 3)
              , MAKE_STRUCT(reverse_sequence_test_params, {16, 1, 1024}, 0, 1)
              , MAKE_STRUCT(reverse_sequence_test_params, {20, 1, 1000}, 0, 1)
              , MAKE_STRUCT(reverse_sequence_test_params, {5, 6, 18}, 2, 2)
              , MAKE_STRUCT(reverse_sequence_test_params, {3, 4, 6}, 2, 1)
              , MAKE_STRUCT(reverse_sequence_test_params, {1,  1, 4, 2, 5}, 3, 4)
              , MAKE_STRUCT(reverse_sequence_test_params, {1,  4, 2, 5}, 2, 3)
              , MAKE_STRUCT(reverse_sequence_test_params, {12, 44, 23, 15}, 0, 3)
              , MAKE_STRUCT(reverse_sequence_test_params, {3, 4, 3, 1}, 2, 3)
              , MAKE_STRUCT(reverse_sequence_test_params, {100, 1, 1, 1}, 0, 3)
              , MAKE_STRUCT(reverse_sequence_test_params, {100, 1, 1, 1}, 0, 2)
              , MAKE_STRUCT(reverse_sequence_test_params, {100, 1, 1, 1}, 0, 0)
              , MAKE_STRUCT(reverse_sequence_test_params, {103, 1, 1, 1}, 0, 0)
              , MAKE_STRUCT(reverse_sequence_test_params, {100, 10, 24}, 0, 0)
              , MAKE_STRUCT(reverse_sequence_test_params, {100, 10, 24}, 0, 1)
              , MAKE_STRUCT(reverse_sequence_test_params, {100, 10, 24}, 0, 2)
              , MAKE_STRUCT(reverse_sequence_test_params, {100, 10, 24}, 1, 2)
              , MAKE_STRUCT(reverse_sequence_test_params, {100, 10, 24}, 1, 1)
        ),
        ::testing::Values<IRVersion>(IRVersion::v7, IRVersion::v10)
    )
);
