// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_pad_test.hpp"

INSTANTIATE_TEST_SUITE_P(accuracy, myriadLayerPad_smoke,
                        ::testing::Combine(
                            ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 64, 16, 16)),
                            ::testing::Values<pad_parameters>(MAKE_STRUCT(pad_parameters, 0, 32, 1, 2,  0, 32, 3, 4)),
                            ::testing::Values<layoutPreference>(vpu::LayoutPreference::ChannelMajor,
                                                                vpu::LayoutPreference::ChannelMinor),
                            ::testing::Values<pad_mode>(std::string("constant"), std::string("edge"), std::string("reflect"), std::string("symmetric")),
                            ::testing::Values<IRVersion>(IRVersion::v7, IRVersion::v10)
                        )
                       );
