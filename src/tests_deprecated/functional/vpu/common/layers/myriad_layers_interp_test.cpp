// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_interp_test.hpp"

INSTANTIATE_TEST_SUITE_P(
        accuracy, myriadLayersTestsInterp_smoke,
        ::testing::Combine(

            ::testing::Values<SizeInputOutput>(
                                                MAKE_STRUCT(interp_test_params, 128, 64, 256, 128, 128),
                                                MAKE_STRUCT(interp_test_params, 128, 64, 256, 128, 128),
                                                MAKE_STRUCT(interp_test_params, 128, 64, 512, 256, 19),
                                                MAKE_STRUCT(interp_test_params, 6,    6, 64, 32, 1024),
                                                MAKE_STRUCT(interp_test_params, 1,    1, 64, 32, 1024)
                                              )

          , ::testing::Values<layoutPreference>(vpu::LayoutPreference::ChannelMajor,
                                                vpu::LayoutPreference::ChannelMinor)
          , ::testing::Values<align_corners>(
                                        true,
                                        false
                                            )
          )
        );
