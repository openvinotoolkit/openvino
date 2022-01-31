// Copyright (C) 2018-2022 Intel Corporation
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

    const std::map<std::string, InterpolateMode, ie::details::CaselessLess<std::string>> interpModeMap = {
        {g_nearest,      InterpolateMode::Nearest},
        {g_linear,       InterpolateMode::Linear},
        {g_linear_onnx,  InterpolateMode::LinearOnnx},
    };

    const std::map<std::string, InterpolateNearestMode, ie::details::CaselessLess<std::string>> nearestModeMap = {
        {g_round_prefer_floor, InterpolateNearestMode::RoundPreferFloor},
        {g_round_prefer_ceil,  InterpolateNearestMode::RoundPreferCeil},
        {g_floor_mode,         InterpolateNearestMode::Floor},
        {g_ceil_mode,          InterpolateNearestMode::Ceil},
        {g_simple,             InterpolateNearestMode::Simple},
    };

    const std::map<std::string, InterpolateCoordTransMode, ie::details::CaselessLess<std::string>> coordTransformModeMap = {
        {g_asymmetric,           InterpolateCoordTransMode::Asymmetric},
        {g_half_pixel,           InterpolateCoordTransMode::HalfPixel},
        {g_pytorch_half_pixel,   InterpolateCoordTransMode::PytorchHalfPixel},
        {g_tf_half_pixel_for_nn, InterpolateCoordTransMode::TfHalfPixelForNn},
        {g_align_corners,        InterpolateCoordTransMode::AlignCorners},
    };
}  // namespace vpu
