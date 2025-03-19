// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_color_i420_base.hpp"

INSTANTIATE_TYPED_TEST_SUITE_P(type_prop_i420_to_rgb, ConvertI420BaseTest, ::testing::Types<ov::op::v8::I420toRGB>);

INSTANTIATE_TYPED_TEST_SUITE_P(type_prop_i420_to_bgr, ConvertI420BaseTest, ::testing::Types<ov::op::v8::I420toBGR>);
