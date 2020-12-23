// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include <vpu/model/stage.hpp>

namespace vpu {
    constexpr auto g_coordinate_transformation_mode = "coordinate_transformation_mode";
    constexpr auto g_mode                 = "mode";
    constexpr auto g_align_corners        = "align_corners";
    constexpr auto g_asymmetric           = "asymmetric";
    constexpr auto g_linear               = "linear";
    constexpr auto g_half_pixel           = "half_pixel";
    constexpr auto g_linear_onnx          = "linear_onnx";
    constexpr auto g_nearest_mode         = "nearest_mode";
    constexpr auto g_pytorch_half_pixel   = "pytorch_half_pixel";
    constexpr auto g_tf_half_pixel_for_nn = "tf_half_pixel_for_nn";
    constexpr auto g_round_prefer_floor   = "round_prefer_floor";
    constexpr auto g_round_prefer_ceil    = "round_prefer_ceil";
    constexpr auto g_floor_mode           = "floor";
    constexpr auto g_ceil_mode            = "ceil";
    constexpr auto g_simple               = "simple";
    constexpr auto g_antialias            = "antialias";
    constexpr auto g_pads_begin           = "pads_begin";
    constexpr auto g_pads_end             = "pads_end";
    constexpr auto g_nearest              = "nearest";
    constexpr auto g_factor               = "factor";
    constexpr auto g_type                 = "type";
}  // namespace vpu
