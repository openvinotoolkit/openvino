// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_color_to_nv12_base.hpp"
#include "openvino/op/rgb_to_nv12.hpp"
#include "openvino/op/bgr_to_nv12.hpp"

INSTANTIATE_TYPED_TEST_SUITE_P(type_prop_rgb_to_nv12, ConvertToNV12BaseTest, ::testing::Types<ov::op::v16::RGBtoNV12>);

INSTANTIATE_TYPED_TEST_SUITE_P(type_prop_bgr_to_nv12, ConvertToNV12BaseTest, ::testing::Types<ov::op::v16::BGRtoNV12>);
