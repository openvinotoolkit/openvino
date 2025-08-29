// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <ostream>

#include "cpu_types.h"
#include "executor_config.hpp"
#include "post_ops.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu {

enum class InterpolateLayoutType : uint8_t { 
    planar, 
    block, 
    by_channel 
};

enum class InterpolateMode : uint8_t { 
    nearest, 
    linear, 
    linear_onnx, 
    cubic, 
    bilinear_pillow, 
    bicubic_pillow 
};

inline std::ostream& operator<<(std::ostream& os, InterpolateMode mode) {
    switch (mode) {
        case InterpolateMode::nearest: os << "nearest"; break;
        case InterpolateMode::linear: os << "linear"; break;
        case InterpolateMode::linear_onnx: os << "linear_onnx"; break;
        case InterpolateMode::cubic: os << "cubic"; break;
        case InterpolateMode::bilinear_pillow: os << "bilinear_pillow"; break;
        case InterpolateMode::bicubic_pillow: os << "bicubic_pillow"; break;
    }
    return os;
}

enum class InterpolateCoordTransMode : uint8_t {
    half_pixel,
    pytorch_half_pixel,
    asymmetric,
    tf_half_pixel_for_nn,
    align_corners
};

enum class InterpolateNearestMode : uint8_t { 
    round_prefer_floor, 
    round_prefer_ceil, 
    floor, 
    ceil, 
    simple 
};

enum class InterpolateShapeCalcMode : uint8_t { 
    sizes, 
    scales 
};

struct InterpolateAttrs {
    // Core interpolation parameters
    InterpolateMode mode = InterpolateMode::nearest;
    InterpolateCoordTransMode coordTransMode = InterpolateCoordTransMode::half_pixel;
    InterpolateNearestMode nearestMode = InterpolateNearestMode::round_prefer_floor;
    InterpolateShapeCalcMode shapeCalcMode = InterpolateShapeCalcMode::sizes;
    InterpolateLayoutType layout = InterpolateLayoutType::planar;
    
    // Interpolation specific attributes
    bool antialias = false;
    float cubeCoeff = -0.75f;
    
    // Padding
    std::vector<int> padBegin;
    std::vector<int> padEnd;
    bool hasPad = false;
    
    // Scale factors for each dimension
    std::vector<float> dataScales;
    
    // Axes to interpolate
    std::vector<int> axes;
    
    // Input/output precisions
    ov::element::Type inPrc = ov::element::f32;
    ov::element::Type outPrc = ov::element::f32;
    
    // Special optimization flag for NHWC as NCHW
    bool NCHWAsNHWC = false;
    
    // Post operations for fusion
    PostOps postOps;
};

using InterpolateConfig = executor::Config<InterpolateAttrs>;

}  // namespace ov::intel_cpu