// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace ov {
namespace intel_cpu {

/**
 * @brief GridSampleDecompositionBilinear decomposes GridSample operation with BILINEAR mode
 * into primitive operations for ARM platforms.
 */
class GridSampleDecompositionBilinear : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("GridSampleDecompositionBilinear", "0");
    GridSampleDecompositionBilinear();
};

/**
 * @brief GridSampleDecompositionNearest decomposes GridSample operation with NEAREST mode
 * into primitive operations for ARM platforms.
 */
class GridSampleDecompositionNearest : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("GridSampleDecompositionNearest", "0");
    GridSampleDecompositionNearest();
};

/**
 * @brief GridSampleDecompositionBicubic decomposes GridSample operation with BICUBIC mode
 * into primitive operations for ARM platforms.
 */
class GridSampleDecompositionBicubic : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("GridSampleDecompositionBicubic", "0");
    GridSampleDecompositionBicubic();
};

/**
 * @brief GridSampleDecomposition is a composite transformation that applies
 * the appropriate decomposition based on the interpolation mode.
 * 
 * The decomposition follows these steps:
 * 1. Denormalize grid coordinates from [-1, 1] to pixel coordinates
 * 2. Find neighboring integer coordinates using Floor (for interpolation modes)
 * 3. Apply appropriate padding mode (zeros, border, reflection)
 * 4. Extract pixel values using GatherND
 * 5. Calculate interpolation weights (for bilinear/bicubic)
 * 6. Compute weighted sum to get final result
 */
class GridSampleDecomposition : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("GridSampleDecomposition", "0");
    GridSampleDecomposition();
};

}  // namespace intel_cpu
}  // namespace ov