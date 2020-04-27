// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "space_to_depth_tests.hpp"

INSTANTIATE_TEST_CASE_P(
        smoke_GPU_TestsSpaceToDepth, SpaceToDepthTests,
        ::testing::Values(
        space_to_depth_test_params{ "GPU", "FP32", { 1, 1, 6, 4 }, "blocks_first", 2, { 1, 4, 3, 2 } },
        space_to_depth_test_params{ "GPU", "FP32", { 1, 1, 9, 9 }, "blocks_first", 3, { 1, 9, 3, 3 } },
        space_to_depth_test_params{ "GPU", "FP32", { 1, 2, 9, 9 }, "blocks_first", 3, { 1, 18, 3, 3 } },
        space_to_depth_test_params{ "GPU", "FP32", { 1, 10, 4096, 1024 }, "blocks_first", 4, { 1, 160, 1024, 256 } },
        space_to_depth_test_params{ "GPU", "FP32", { 1, 1, 6, 4 }, "depth_first", 2, { 1, 4, 3, 2 } },
        space_to_depth_test_params{ "GPU", "FP32", { 1, 1, 9, 9 }, "depth_first", 3, { 1, 9, 3, 3 } },
        space_to_depth_test_params{ "GPU", "FP32", { 1, 2, 9, 9 }, "depth_first", 3, { 1, 18, 3, 3 } },
        space_to_depth_test_params{ "GPU", "FP32", { 1, 10, 4096, 1024 }, "depth_first", 4, { 1, 160, 1024, 256 } }
));
