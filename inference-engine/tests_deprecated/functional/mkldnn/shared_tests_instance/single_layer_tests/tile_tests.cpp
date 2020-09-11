// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tile_tests.hpp"

tile_test_params tile_test_cases[] = {
        tile_test_params("CPU", case_1),
        tile_test_params("CPU", case_2),
        tile_test_params("CPU", case_3),
        tile_test_params("CPU", case_4),
        tile_test_params("CPU", case_5),
        tile_test_params("CPU", case_6),
        tile_test_params("CPU", case_7),
        tile_test_params("CPU", case_8),
        tile_test_params("CPU", case_9),
        tile_test_params("CPU", case_10),
        tile_test_params("CPU", case_11),
        tile_test_params("CPU", case_12),
        tile_test_params("CPU", case_13),
        tile_test_params("CPU", case_14),
        tile_test_params("CPU", case_15),
        tile_test_params("CPU", case_16),
};

INSTANTIATE_TEST_CASE_P(smoke_CPU_TestsGeneralTile, TileTest, ::testing::ValuesIn(tile_test_cases));
