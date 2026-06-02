// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "io_layouts.hpp"

// TODO: check if we need more test cases here
// are **SCALAR**, '?', and '...' supported?
// see src/core/src/layout.cpp
INSTANTIATE_TEST_SUITE_P(
    IOLayoutsSectionUnitTests,
    ValidLayouts,
    ::testing::Values(
        std::make_tuple(std::vector<ov::Layout>{ov::Layout("NCHW")}, std::vector<ov::Layout>{ov::Layout("NCHW")}),
        std::make_tuple(std::vector<ov::Layout>{ov::Layout("NHWC")}, std::vector<ov::Layout>{ov::Layout("NHWC")}),
        std::make_tuple(std::vector<ov::Layout>{ov::Layout("NC")}, std::vector<ov::Layout>{ov::Layout("NC")}),
        std::make_tuple(std::vector<ov::Layout>{ov::Layout("NH")}, std::vector<ov::Layout>{ov::Layout("NH")}),
        std::make_tuple(std::vector<ov::Layout>{ov::Layout("NHW")}, std::vector<ov::Layout>{ov::Layout("NHW")}),
        std::make_tuple(std::vector<ov::Layout>{ov::Layout("NWH")}, std::vector<ov::Layout>{ov::Layout("NWH")}),
        std::make_tuple(std::vector<ov::Layout>{ov::Layout("NCDHW")}, std::vector<ov::Layout>{ov::Layout("NDHWC")}),
        std::make_tuple(std::vector<ov::Layout>{ov::Layout("NCHW"), ov::Layout("NHWC")},
                        std::vector<ov::Layout>{ov::Layout("NCHW"), ov::Layout("NHWC")}),
        std::make_tuple(std::vector<ov::Layout>{ov::Layout("NCHW"), ov::Layout("NHW")},
                        std::vector<ov::Layout>{ov::Layout("NC")}),
        std::make_tuple(std::vector<ov::Layout>{ov::Layout("NCHW"),
                                                ov::Layout("NHWC"),
                                                ov::Layout("NHW"),
                                                ov::Layout("NWH"),
                                                ov::Layout("NC"),
                                                ov::Layout("NH"),
                                                ov::Layout("NCDHW")},
                        std::vector<ov::Layout>{ov::Layout("NCHW"),
                                                ov::Layout("NHWC"),
                                                ov::Layout("NHW"),
                                                ov::Layout("NWH"),
                                                ov::Layout("NC"),
                                                ov::Layout("NH"),
                                                ov::Layout("NDHWC")}),
        std::make_tuple(std::vector<ov::Layout>{}, std::vector<ov::Layout>{ov::Layout("NCHW")}),
        std::make_tuple(std::vector<ov::Layout>{ov::Layout("NCHW")}, std::vector<ov::Layout>{}),
        std::make_tuple(std::vector<ov::Layout>{}, std::vector<ov::Layout>{})));
