// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace ov {
namespace intel_cpu {

/**
 * @brief GridSampleDecomposition decomposes GridSample operation into primitive operations
 * for ARM platforms which don't have optimized ACL implementation.
 * 
 * Currently supports only bilinear interpolation with border padding mode.
 * 
 * The decomposition follows these steps:
 * 1. Denormalize grid coordinates from [-1, 1] to pixel coordinates
 * 2. Find neighboring integer coordinates using Floor
 * 3. Clamp coordinates to image boundaries
 * 4. Extract pixel values using GatherND
 * 5. Calculate interpolation weights
 * 6. Compute weighted sum to get final result
 */
class GridSampleDecomposition : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("GridSampleDecomposition", "0");
    GridSampleDecomposition();
};

}  // namespace intel_cpu
}  // namespace ov