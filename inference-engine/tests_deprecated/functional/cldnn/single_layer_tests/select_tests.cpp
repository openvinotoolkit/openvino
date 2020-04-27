// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "select_tests.hpp"

TEST_P(SelectTests, smoke_GPU_TestsSelectNoneBroadcast) {}

INSTANTIATE_TEST_CASE_P(
    smoke_TestsSelectNoneBroadcast, SelectTests,
    ::testing::Values(
          select_params{ "GPU", {1}, {1}, {1}, "none", false },
          select_params{ "GPU", {17}, {17}, {17}, "none", false  },
          select_params{ "GPU", {33, 35}, {33, 35}, {33, 35}, "none", false  },
          select_params{ "GPU", {6, 7, 8}, {6, 7, 8}, {6, 7, 8}, "none", false  },
          select_params{ "GPU", {2, 3, 4, 5}, {2, 3, 4, 5}, {2, 3, 4, 5}, "none", false  },
          select_params{ "GPU", {3, 24, 35, 9}, {3, 24, 35, 9}, {3, 24, 35, 9}, "none", false  },
          select_params{ "GPU", {8, 14, 32, 12}, {8, 14, 32, 12}, {8, 14, 32, 12}, "none", false  },
          select_params{ "GPU", {16, 32, 15, 54}, {16, 32, 15, 54}, {16, 32, 15, 54}, "none", false  }
));

INSTANTIATE_TEST_CASE_P(
    smoke_TestsSelectNumpyBroadcast, SelectTests,
    ::testing::Values(
          select_params{ "GPU", {1}, {1}, {1}, "numpy", false },
          select_params{ "GPU", {17}, {17}, {17}, "numpy", false },
          select_params{ "GPU", {33, 35}, {33, 35}, {33, 35}, "numpy", false },
          select_params{ "GPU", {6, 7, 8}, {6, 7, 8}, {6, 7, 8}, "numpy", false },
          select_params{ "GPU", {2, 3, 4, 5}, {2, 3, 4, 5}, {2, 3, 4, 5}, "numpy", false },
          select_params{ "GPU", {3, 24, 35, 9}, {3, 24, 35, 9}, {3, 24, 35, 9}, "numpy", false },
          select_params{ "GPU", {8, 14, 32, 12}, {8, 14, 32, 12}, {8, 14, 32, 12}, "numpy", false },
          select_params{ "GPU", {16, 32, 15, 54}, {16, 32, 15, 54}, {16, 32, 15, 54}, "numpy", false },

          select_params{ "GPU", {17}, {1}, {17}, "numpy", false },
          select_params{ "GPU", {1}, {17}, {17}, "numpy", false },
          select_params{ "GPU", {17}, {17}, {1}, "numpy", false },
          select_params{ "GPU", {17}, {1}, {1}, "numpy", false },
          select_params{ "GPU", {1}, {17}, {1}, "numpy", false },
          select_params{ "GPU", {33, 1}, {33, 35}, {33, 35}, "numpy", false },
          select_params{ "GPU", {33, 35}, {33, 35}, {35}, "numpy", false },
          select_params{ "GPU", {33, 35}, {33, 35}, {1}, "numpy", false },
          select_params{ "GPU", {35}, {33, 1}, {35}, "numpy", false },
          select_params{ "GPU", {35, 9}, {24, 35, 9}, {24, 35, 9}, "numpy", false },
          select_params{ "GPU", {24, 35, 9}, {24, 35, 9}, {35, 9}, "numpy", false },
          select_params{ "GPU", {9}, {24, 35, 1}, {35, 9}, "numpy", false },
          select_params{ "GPU", {24, 35, 1}, {35, 9}, {24, 35, 1}, "numpy", false },
          select_params{ "GPU", {24, 1, 9}, {9}, {24, 1, 9}, "numpy", false },
          select_params{ "GPU", {24, 1, 9}, {24, 35, 1}, {1}, "numpy", false },
          select_params{ "GPU", {24, 35, 9}, {24, 35, 9}, {24, 1, 9}, "numpy", false },
          select_params{ "GPU", {24, 1, 9}, {24, 35, 1}, {24, 35, 9}, "numpy", false },
          select_params{ "GPU", {16, 32, 15, 54}, {16, 1, 15, 54}, {16, 32, 15, 54}, "numpy", false },
          select_params{ "GPU", {1}, {16, 32, 15, 54}, {16, 32, 1, 54}, "numpy", false },
          select_params{ "GPU", {3, 24, 35, 9}, {24, 35, 9}, {3, 1, 35, 9}, "numpy", false },
          select_params{ "GPU", {3, 24, 35, 9}, {9}, {3, 24, 35, 9}, "numpy", false },
          select_params{ "GPU", {16, 1, 15, 54}, {16, 32, 15, 54}, {16, 32, 1, 54}, "numpy", false },
          select_params{ "GPU", {16, 32, 1, 1}, {16, 32, 15, 54}, {16, 32, 15, 54}, "numpy", false },
          select_params{ "GPU", {8, 14, 32, 1}, {8, 14, 32, 12}, {32, 12}, "numpy", false },
          select_params{ "GPU", {16, 32, 15, 54}, {16, 1, 1, 1}, {16, 32, 1, 54}, "numpy", false },
          select_params{ "GPU", {16, 1, 15, 54}, {16, 32, 1, 54}, {16, 32, 15, 1}, "numpy", false },
          select_params{ "GPU", {35, 9}, {3, 24, 1, 1}, {3, 24, 35, 9}, "numpy", false },
          select_params{ "GPU", {3, 24, 1, 1}, {35, 9}, {35, 9}, "numpy", false },
          select_params{ "GPU", {9}, {3, 1, 1, 1}, {3, 1, 1, 1}, "numpy", false }
));

INSTANTIATE_TEST_CASE_P(
    smoke_TestsSelectNoneBroadcastError, SelectTests,
    ::testing::Values(
          select_params{ "GPU", {1, 32, 15, 54}, {1, 32, 15, 54}, {16, 32, 15, 54}, "none", true },
          select_params{ "GPU", {16, 1, 15, 54}, {16, 1, 15, 54}, {16, 32, 15, 54}, "none", true },
          select_params{ "GPU", {16, 32, 15, 54}, {16, 32, 15, 54}, {16, 32, 16, 54}, "none", true },
          select_params{ "GPU", {16, 32, 15, 1}, {16, 32, 15, 1}, {16, 32, 15, 54}, "none", true },
          select_params{ "GPU", {15, 32, 15, 54}, {16, 32, 15, 54}, {15, 32, 15, 54}, "none", true },
          select_params{ "GPU", {16, 33, 15, 54}, {16, 32, 15, 54}, {16, 32, 15, 54}, "none", true },
          select_params{ "GPU", {16, 32, 16, 54}, {16, 32, 15, 54}, {16, 32, 16, 54}, "none", true },
          select_params{ "GPU", {16, 32, 15, 54}, {16, 32, 15, 54}, {16, 32, 15, 56}, "none", true },
          select_params{ "GPU", {3, 5, 35, 9}, {3, 24, 35, 7}, {3, 24, 35, 9}, "none", true },
          select_params{ "GPU", {11, 24, 35, 9}, {3, 24, 35, 9}, {3, 24, 7, 9}, "none", true },
          select_params{ "GPU", {3, 24, 35, 9}, {3, 24, 35, 9}, {3, 24, 35, 9}, "none", true },
          select_params{ "GPU", {11, 24, 35, 11}, {7, 13, 35, 9}, {3, 24, 27, 17}, "none", true },
          select_params{ "GPU", {1}, {1}, {9}, "none", true },

          select_params{ "GPU", {32, 15, 54}, {16, 32, 15, 54}, {15, 32, 15, 54}, "none", true },
          select_params{ "GPU", {16, 32, 15, 54}, {16, 1, 15, 54}, {16, 33, 15, 54}, "none", true },
          select_params{ "GPU", {16, 32, 1, 54}, {16, 32, 15, 1}, {16, 32, 2, 3}, "none", true },
          select_params{ "GPU", {7, 1, 14}, {7, 14, 14}, {7, 7, 14, 14}, "none", true },
          select_params{ "GPU", {7, 1, 14}, {7, 14, 14}, {7, 1, 1, 14}, "none", true },
          select_params{ "GPU", {35, 9}, {35, 1}, {24, 35, 9}, "none", true },
          select_params{ "GPU", {1}, {9}, {35, 9}, "none", true },
        
          select_params{ "GPU", {17}, {1}, {17}, "none", true },
          select_params{ "GPU", {1}, {17}, {17}, "none", true },
          select_params{ "GPU", {17}, {17}, {1}, "none", true },
          select_params{ "GPU", {17}, {1}, {1}, "none", true },
          select_params{ "GPU", {1}, {17}, {1}, "none", true },
          select_params{ "GPU", {33, 1}, {33, 35}, {33, 35}, "none", true },
          select_params{ "GPU", {33, 35}, {33, 35}, {35}, "none", true },
          select_params{ "GPU", {33, 35}, {33, 35}, {1}, "none", true },
          select_params{ "GPU", {35}, {33, 1}, {35}, "none", true },
          select_params{ "GPU", {35, 9}, {24, 35, 9}, {24, 35, 9}, "none", true },
          select_params{ "GPU", {24, 35, 9}, {24, 35, 9}, {35, 9}, "none", true },
          select_params{ "GPU", {9}, {24, 35, 1}, {35, 9}, "none", true },
          select_params{ "GPU", {24, 35, 1}, {35, 9}, {24, 35, 1}, "none", true },
          select_params{ "GPU", {24, 1, 9}, {9}, {24, 1, 9}, "none", true },
          select_params{ "GPU", {24, 1, 9}, {24, 35, 1}, {1}, "none", true },
          select_params{ "GPU", {24, 35, 9}, {24, 35, 9}, {24, 1, 9}, "none", true },
          select_params{ "GPU", {24, 1, 9}, {24, 35, 1}, {24, 35, 9}, "none", true },
          select_params{ "GPU", {16, 32, 15, 54}, {16, 1, 15, 54}, {16, 32, 15, 54}, "none", true },
          select_params{ "GPU", {1}, {16, 32, 15, 54}, {16, 32, 1, 54}, "none", true },
          select_params{ "GPU", {3, 24, 35, 9}, {24, 35, 9}, {3, 1, 35, 9}, "none", true },
          select_params{ "GPU", {3, 24, 35, 9}, {9}, {3, 24, 35, 9}, "none", true },
          select_params{ "GPU", {16, 1, 15, 54}, {16, 32, 15, 54}, {16, 32, 1, 54}, "none", true },
          select_params{ "GPU", {16, 32, 1, 1}, {16, 32, 15, 54}, {16, 32, 15, 54}, "none", true },
          select_params{ "GPU", {8, 14, 32, 1}, {8, 14, 32, 12}, {32, 12}, "none", true },
          select_params{ "GPU", {16, 32, 15, 54}, {16, 1, 1, 1}, {16, 32, 1, 54}, "none", true },
          select_params{ "GPU", {16, 1, 15, 54}, {16, 32, 1, 54}, {16, 32, 15, 1}, "none", true },
          select_params{ "GPU", {35, 9}, {3, 24, 1, 1}, {3, 24, 35, 9}, "none", true },
          select_params{ "GPU", {3, 24, 1, 1}, {35, 9}, {35, 9}, "none", true },
          select_params{ "GPU", {9}, {3, 1, 1, 1}, {3, 1, 1, 1}, "none", true }
));

INSTANTIATE_TEST_CASE_P(
    smoke_TestsSelectNumpyBroadcastError, SelectTests,
    ::testing::Values(
          select_params{ "GPU", {1, 32, 15, 54}, {1, 32, 15, 54}, {16, 32, 15, 54}, "numpy", true },
          select_params{ "GPU", {16, 1, 15, 54}, {16, 1, 15, 54}, {16, 32, 15, 54}, "numpy", true },
          select_params{ "GPU", {16, 32, 15, 54}, {16, 32, 15, 54}, {16, 32, 16, 54}, "numpy", true },
          select_params{ "GPU", {16, 32, 15, 1}, {16, 32, 15, 1}, {16, 32, 15, 54}, "numpy", true },
          select_params{ "GPU", {15, 32, 15, 54}, {16, 32, 15, 54}, {15, 32, 15, 54}, "numpy", true },
          select_params{ "GPU", {16, 33, 15, 54}, {16, 32, 15, 54}, {16, 32, 15, 54}, "numpy", true },
          select_params{ "GPU", {16, 32, 16, 54}, {16, 32, 15, 54}, {16, 32, 16, 54}, "numpy", true },
          select_params{ "GPU", {16, 32, 15, 54}, {16, 32, 15, 54}, {16, 32, 15, 56}, "numpy", true },
          select_params{ "GPU", {3, 5, 35, 9}, {3, 24, 35, 7}, {3, 24, 35, 9}, "numpy", true },
          select_params{ "GPU", {11, 24, 35, 9}, {3, 24, 35, 9}, {3, 24, 7, 9}, "numpy", true },
          select_params{ "GPU", {3, 24, 35, 9}, {3, 24, 35, 9}, {3, 24, 35, 9}, "numpy", true },
          select_params{ "GPU", {11, 24, 35, 11}, {7, 13, 35, 9}, {3, 24, 27, 17}, "numpy", true },
          select_params{ "GPU", {1}, {1}, {9}, "numpy", true },

          select_params{ "GPU", {32, 15, 54}, {16, 32, 15, 54}, {15, 32, 15, 54}, "numpy", true },
          select_params{ "GPU", {16, 32, 15, 54}, {16, 1, 15, 54}, {16, 33, 15, 54}, "numpy", true },
          select_params{ "GPU", {16, 32, 1, 54}, {16, 32, 15, 1}, {16, 32, 2, 3}, "numpy", true },
          select_params{ "GPU", {7, 1, 14}, {7, 14, 14}, {7, 7, 14, 14}, "numpy", true },
          select_params{ "GPU", {7, 1, 14}, {7, 14, 14}, {7, 1, 1, 14}, "numpy", true },
          select_params{ "GPU", {35, 9}, {35, 1}, {24, 35, 9}, "numpy", true },
          select_params{ "GPU", {1}, {9}, {35, 9}, "numpy", true }
));
