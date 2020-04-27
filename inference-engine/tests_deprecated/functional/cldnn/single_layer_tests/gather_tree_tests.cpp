// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gather_tree_tests.hpp"

INSTANTIATE_TEST_CASE_P(
    smoke_GPU_TestsGatherTree, GatherTreeTests,
    ::testing::Values(
        // Params: in_out_shape, step_idx, parent_idx, max_seq_len, end_token, reference
        gather_tree_test_params{ {3, 2, 3 }, {1, 2, 3, 2, 3, 4, 4, 5, 6, 5, 6, 7, 7, 8, 9, 8, 9, 10}, {0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 0, 2, 1, 2, 2, 1, 1},
                                  {3, 3 }, {11}, {2, 2, 2, 2, 4, 4, 6, 5, 6, 7, 6, 6, 7, 8, 9, 8, 9, 10}, "GPU"},
        gather_tree_test_params{ {4, 1, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, -1, -1, -1}, {0, 0, 0, 0, 1, 1, 2, 1, 2, -1, -1, -1},
                                  {3}, {10}, {2, 2, 2, 6, 5, 6, 7, 8, 9, 10, 10, 10}, "GPU"},
        gather_tree_test_params{ {4, 1, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10}, {0, 0, 0, 0, 1, 1, 2, 1, 2, 1, 1, 1},
                                  {4}, {10}, {2, 2, 2, 5, 5, 5, 8, 8, 8, 10, 10, 10}, "GPU"},
        gather_tree_test_params{ {5, 1, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 10, 3, 2, 10, 10}, {0, 0, 0, 0, 1, 1, 2, 1, 2, 1, 1, 1, 2, 0, 1},
                                  {5}, {10}, {2, 2, 2, 5, 5, 5, 8, 8, 8, 3, 1, 10, 2, 10, 10}, "GPU"},
        gather_tree_test_params{ {4, 2, 3}, {1, 2, 3, 2, 3, 4, 4, 5, 6, 5, 6, 7, 7, 8, 9, 8, 9, 10, 0, 0, 0, 11, 12, 0},
                                  {0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 2, 1, 2, 2, 0, 1, -1, -1, -1, 0, 1, 0},
                                  {3, 4}, {11}, {2, 2, 2, 2, 3, 2, 6, 5, 6, 7, 5, 7, 7, 8, 9, 8, 9, 8, 11, 11, 11, 11, 12, 0}, "GPU"}
));
