// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_topk_test.hpp"

static const std::vector<Geometry> s_geometries_list =
{
    {{ 1, 4 }, 1, 3 },
    {{ 3, 4, 7, 5, 6 }, 1, 2 },
    {{ 5, 6, 3, 4 }, 2, 2 },
// TODO: 3D geometries excluded due to incorrect CHW/HWC layouts processing in IE/GT; uncomment when fixed
//    {{ 223, 217, 21 }, 0, 13 },
//    {{ 439, 429, 5 }, 2, 2 },
    {{ 65, 33 }, 1, 3 },
    {{ 31680, 1 }, 0, 13 },
    {{ 495, 1 }, 0, 7 },
    {{ 80000 }, 0, 117 },
    {{ 3639 }, 0, 3 },
};

static const std::vector<std::string> s_modes_list =
{
    "max",
    "min",
};

static const std::vector<std::string> s_sorts_list =
{
    "value",
    "index",
//    "none", // currently is not supported by firmware
};

INSTANTIATE_TEST_SUITE_P(accuracy, myriadTestsTopK_smoke,
    ::testing::Combine(
        ::testing::ValuesIn(s_geometries_list),
        ::testing::ValuesIn(s_modes_list),
        ::testing::ValuesIn(s_sorts_list))
);
