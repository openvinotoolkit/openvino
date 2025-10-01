// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/matcher_pass.hpp"

namespace ov::intel_cpu {

/**
 * @brief GridSampleDecomposition decomposes GridSample operation into primitive operations
 *
 * This transformation enables execution on ARM platforms without native GridSample support
 * by decomposing the operation into simpler primitives that can be optimized by ACL.
 *
 * The transformation handles all three interpolation modes:
 * - NEAREST: Simpler mode that finds the nearest pixel without interpolation
 * - BILINEAR: Uses 2x2 pixel neighborhood with bilinear weighting
 * - BICUBIC: Uses 4x4 pixel neighborhood with cubic weighting for smoother results
 *
 * Common decomposition pattern for all modes:
 *
 * Before:
 *    +-------------------+    +----------------------+
 *    | Data[N,C,H,W]     |    | Grid[N,H_out,W_out,2]|
 *    +--------+----------+    +---------+------------+
 *             |                         |
 *        +----v-------------------------v----+
 *        | GridSample                        |
 *        | (mode, padding, align_corners)    |
 *        +----------------+------------------+
 *                         |
 *                +--------v---------------+
 *                | Output[N,C,H_out,W_out]|
 *                +------------------------+
 *
 * After (general structure):
 *    +-------------------+    +----------------------+
 *    | Data[N,C,H,W]     |    | Grid[N,H_out,W_out,2]|
 *    +--------+----------+    +---------+------------+
 *             |                         |
 *             |                   +-----v------+
 *             |                   | Split      |
 *             |                   | x,y coords |
 *             |                   +--+-----+---+
 *             |                      |     |
 *             |                 +----v-+  +v----+
 *             |                 | Norm |  | Norm|
 *             |                 | X    |  | Y   |
 *             |                 +----+-+  +-+---+
 *             |                      |      |
 *             |                 +----v------v---+
 *             |                 | Mode-specific |
 *             |                 | interpolation |
 *             |                 +----+----------+
 *             |                      |
 *        +----v----------------------v----+
 *        | GatherND (mode-specific)       |
 *        +----------------+---------------+
 *                         |
 *                   +-----v--------+
 *                   | Weighted     |
 *                   | combination  |
 *                   +-----+--------+
 *                         |
 *                +--------v---------------+
 *                | Output[N,C,H_out,W_out]|
 *                +------------------------+
 *
 * The matcher selects the appropriate decomposition path based on
 * ov::op::v9::GridSample::Attributes::mode and applies special-case handling
 * for known problematic combinations (e.g., non-f32 inputs, specific padding modes).
 */
class GridSampleDecomposition : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("GridSampleDecomposition", "0");
    GridSampleDecomposition();
};

}  // namespace ov::intel_cpu