// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_gemm_test.hpp"

INSTANTIATE_TEST_SUITE_P(accuracy, myriadLayerGEMM_smoke,
        ::testing::Combine(
        ::testing::Values<gemm_parameters>(
                MAKE_STRUCT(gemm_parameters, 4.7f, 2.3f, 5,  7,   11,  1, 2,   3, 4,  5, 6,  7, 8),
                MAKE_STRUCT(gemm_parameters, 1.0f, 1.0f, 1, 16, 1024, 10, 1,  10, 1, 10, 1, 10, 1),
                MAKE_STRUCT(gemm_parameters, 1.0f, 1.0f, 3,  5,    6,  1, 1,   1, 1,  1, 1,  1, 1),

                MAKE_STRUCT(gemm_parameters, 1.0f, 1.0f,   8, 17,   32,  1, 12, 1, 12,  1, 12, 1, 12),
                MAKE_STRUCT(gemm_parameters, 1.0f, 1.0f, 128, 128, 128,  1, 12, 1, 12,  1, 12, 1, 12),
                MAKE_STRUCT(gemm_parameters, 1.0f, 1.0f, 128, 768, 768,  1, 1,  1,  1,  1,  1, 1, 1),
                MAKE_STRUCT(gemm_parameters, 1.0f, 1.0f, 128, 768, 3072, 1, 1,  1,  1,  1,  1, 1, 1),
                MAKE_STRUCT(gemm_parameters, 1.0f, 1.0f, 128, 768, 3072, 1, 2,  1,  2,  1,  2, 1, 2),

                MAKE_STRUCT(gemm_parameters, 1.0f, 1.0f, 8 * 1, 5, 8 * 7,  1, 1,  1, 1,  1, 1, 1, 1)
        ),

        ::testing::Values<layoutPreference>(vpu::LayoutPreference::ChannelMajor),
        ::testing::Values<hasThreeInputs>(true, false),
        ::testing::Values<transposeA>(true, false),
        ::testing::Values<transposeB>(true, false)
        )
);
