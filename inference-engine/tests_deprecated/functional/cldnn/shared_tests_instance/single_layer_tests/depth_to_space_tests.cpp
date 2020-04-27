// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "depth_to_space_tests.hpp"

INSTANTIATE_TEST_CASE_P(
        smoke_GPU_TestsDepthToSpace, DepthToSpaceTests,
        ::testing::Values(
        depth_to_space_test_params{ "GPU", "FP32", { 1, 4, 1, 1 }, 2, { 1, 1, 2, 2 } },
        depth_to_space_test_params{ "GPU", "FP32", { 1, 4, 2, 1 }, 2, { 1, 1, 4, 2 } },
        depth_to_space_test_params{ "GPU", "FP32", { 1, 4, 2, 2 }, 2, { 1, 1, 4, 4 } },
        depth_to_space_test_params{ "GPU", "FP32", { 1, 4, 3, 2 }, 2, { 1, 1, 6, 4 } },
        depth_to_space_test_params{ "GPU", "FP32", { 1, 9, 3, 3 }, 3, { 1, 1, 9, 9 } },
        depth_to_space_test_params{ "GPU", "FP32", { 1, 18, 3, 3 }, 3, { 1, 2, 9, 9 } },
        depth_to_space_test_params{ "GPU", "FP32", { 1, 4, 2048, 512 }, 2, { 1, 1, 4096, 1024 } }
));
