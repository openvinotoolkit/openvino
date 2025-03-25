// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_color_nv12_base.hpp"

INSTANTIATE_TYPED_TEST_SUITE_P(type_prop_nv12_to_rgb, ConvertNV12BaseTest, ::testing::Types<ov::op::v8::NV12toRGB>);

INSTANTIATE_TYPED_TEST_SUITE_P(type_prop_nv12_to_bgr, ConvertNV12BaseTest, ::testing::Types<ov::op::v8::NV12toBGR>);
