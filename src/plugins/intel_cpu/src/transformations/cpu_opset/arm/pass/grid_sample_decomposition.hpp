// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
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
 * @brief GridSampleDecomposition is a composite transformation that registers
 * all three GridSample decomposition passes (Nearest, Bilinear, Bicubic).
 *
 * This pass registers the following transformations:
 * - GridSampleDecompositionNearest
 * - GridSampleDecompositionBilinear
 * - GridSampleDecompositionBicubic
 */
class GridSampleDecomposition : public ov::pass::GraphRewrite {
public:
    OPENVINO_RTTI("GridSampleDecomposition", "0");
    GridSampleDecomposition();
};

}  // namespace intel_cpu
}  // namespace ov