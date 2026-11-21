// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "openvino/op/util/interpolate_base.hpp"

namespace ov {
namespace pass {
namespace interpolate_gather {

/// Mirrors ov::reference::InterpolateEvalHelper::get_input_coords_for_nearest_mode.
inline float source_coordinate(ov::op::util::InterpolateBase::CoordinateTransformMode mode,
                               float x,
                               float scale,
                               int64_t out,
                               int64_t in) {
    switch (mode) {
    case ov::op::util::InterpolateBase::CoordinateTransformMode::PYTORCH_HALF_PIXEL:
        return out > 1 ? (x + 0.5f) / scale - 0.5f : 0.0f;
    case ov::op::util::InterpolateBase::CoordinateTransformMode::ASYMMETRIC:
        return x / scale;
    case ov::op::util::InterpolateBase::CoordinateTransformMode::TF_HALF_PIXEL_FOR_NN:
        return (x + 0.5f) / scale;
    case ov::op::util::InterpolateBase::CoordinateTransformMode::ALIGN_CORNERS:
        return out == 1 ? 0.0f : x * static_cast<float>(in - 1) / static_cast<float>(out - 1);
    case ov::op::util::InterpolateBase::CoordinateTransformMode::HALF_PIXEL:
    default:
        return ((x + 0.5f) / scale) - 0.5f;
    }
}

/// Mirrors ov::reference::get_func(NearestMode).
inline int64_t nearest_pixel(ov::op::util::InterpolateBase::NearestMode mode, float x, bool is_downsample) {
    using NearestMode = ov::op::util::InterpolateBase::NearestMode;
    switch (mode) {
    case NearestMode::ROUND_PREFER_CEIL:
        return static_cast<int64_t>(std::round(x));
    case NearestMode::FLOOR:
        return static_cast<int64_t>(std::floor(x));
    case NearestMode::CEIL:
        return static_cast<int64_t>(std::ceil(x));
    case NearestMode::SIMPLE:
        return is_downsample ? static_cast<int64_t>(std::ceil(x)) : static_cast<int64_t>(x);
    case NearestMode::ROUND_PREFER_FLOOR:
    default:
        if (x == static_cast<int64_t>(x) + 0.5f) {
            return static_cast<int64_t>(std::floor(x));
        }
        return static_cast<int64_t>(std::round(x));
    }
}

inline bool has_non_zero_padding(const std::vector<size_t>& pads) {
    return std::any_of(pads.begin(), pads.end(), [](size_t pad) {
        return pad != 0;
    });
}

}  // namespace interpolate_gather
}  // namespace pass
}  // namespace ov
