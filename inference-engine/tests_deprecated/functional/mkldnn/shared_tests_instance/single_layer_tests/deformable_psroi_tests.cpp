// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "deformable_psroi_tests.hpp"

INSTANTIATE_TEST_CASE_P(
        smoke_TestDeformable, DeformablePSROIOnlyTest,
        ::testing::Values(
                deformable_psroi_test_params{"CPU", {1, 7938, 38, 38}, {300, 5}, {300, 162, 7, 7},
                                             0.0625, 162, 7, 7, 7, 7, 4, true
                },
                deformable_psroi_test_params{"CPU", {1, 392, 38, 38}, {300, 5}, {300, 8, 7, 7},
                                             0.0625, 8, 7, 7, 7, 7, 4, false, 0.1, {300, 2, 7, 7}
                },
                deformable_psroi_test_params{"CPU", {1, 98, 38, 38}, {300, 5}, {300, 2, 7, 7},
                                             0.0625, 2, 7, 7, 7, 7, 4, true
                },
                deformable_psroi_test_params{"CPU", {1, 3969, 38, 38}, {300, 5}, {300, 81, 7, 7},
                                             0.0625, 81, 7, 7, 7, 7, 4, false, 0.1, {300, 162, 7, 7}
                }
        ));
