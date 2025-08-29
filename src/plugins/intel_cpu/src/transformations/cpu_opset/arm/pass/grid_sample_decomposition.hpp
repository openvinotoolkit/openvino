// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/matcher_pass.hpp"

namespace ov::intel_cpu {

/**
 * @brief Decompose GridSample with BILINEAR interpolation into primitive operations
 *
 * Before:
 *    +-------------------+    +----------------------+
 *    | Data[N,C,H,W]     |    | Grid[N,H_out,W_out,2]|
 *    +--------+----------+    +---------+------------+
 *             |                         |
 *        +----v-------------------------v----+
 *        | GridSample                        |
 *        | (mode=bilinear, padding, align)   |
 *        +----------------+------------------+
 *                         |
 *                +--------v---------------+
 *                | Output[N,C,H_out,W_out]|
 *                +------------------------+
 *
 * After:
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
 *             |                 | Floor & Clip  |
 *             |                 | (x0,y0,x1,y1) |
 *             |                 +----+----------+
 *             |                      |
 *             |                 +----v----------+
 *             |                 | Calc weights  |
 *             |                 | (wa,wb,wc,wd) |
 *             |                 +----+----------+
 *             |                      |
 *        +----v----------------------v----+
 *        | GatherND (4 corner pixels)     |
 *        +----------------+---------------+
 *                         |
 *                   +-----v--------+
 *                   | Multiply &   |
 *                   | Add (interp) |
 *                   +-----+--------+
 *                         |
 *                +--------v---------------+
 *                | Output[N,C,H_out,W_out]|
 *                +------------------------+
 *
 * This transformation enables execution on ARM platforms without native GridSample support
 * by decomposing the operation into simpler primitives that can be optimized by ACL.
 */
class GridSampleDecompositionBilinear : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("GridSampleDecompositionBilinear", "0");
    GridSampleDecompositionBilinear();
};

/**
 * @brief Decompose GridSample with NEAREST interpolation into primitive operations
 *
 * Before:
 *    +-------------------+    +----------------------+
 *    | Data[N,C,H,W]     |    | Grid[N,H_out,W_out,2]|
 *    +--------+----------+    +---------+------------+
 *             |                         |
 *        +----v-------------------------v----+
 *        | GridSample                        |
 *        | (mode=nearest, padding, align)    |
 *        +----------------+------------------+
 *                         |
 *                +--------v---------------+
 *                | Output[N,C,H_out,W_out]|
 *                +------------------------+
 *
 * After:
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
 *             |                 | Round         |
 *             |                 | (nearest idx) |
 *             |                 +----+----------+
 *             |                      |
 *             |                 +----v----------+
 *             |                 | Clip          |
 *             |                 | (boundaries)  |
 *             |                 +----+----------+
 *             |                      |
 *        +----v----------------------v----+
 *        | GatherND                       |
 *        +----------------+---------------+
 *                         |
 *                +--------v---------------+
 *                | Output[N,C,H_out,W_out]|
 *                +------------------------+
 *
 * This transformation is simpler than bilinear as it only requires finding
 * the nearest pixel without interpolation between neighboring pixels.
 */
class GridSampleDecompositionNearest : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("GridSampleDecompositionNearest", "0");
    GridSampleDecompositionNearest();
};

/**
 * @brief Decompose GridSample with BICUBIC interpolation into primitive operations
 *
 * Before:
 *    +-------------------+    +----------------------+
 *    | Data[N,C,H,W]     |    | Grid[N,H_out,W_out,2]|
 *    +--------+----------+    +---------+------------+
 *             |                         |
 *        +----v-------------------------v----+
 *        | GridSample                        |
 *        | (mode=bicubic, padding, align)    |
 *        +----------------+------------------+
 *                         |
 *                +--------v---------------+
 *                | Output[N,C,H_out,W_out]|
 *                +------------------------+
 *
 * After:
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
 *             |                 | Get 4x4       |
 *             |                 | neighborhood  |
 *             |                 +----+----------+
 *             |                      |
 *             |                 +----v----------+
 *             |                 | Cubic weights |
 *             |                 | calculation   |
 *             |                 +----+----------+
 *             |                      |
 *        +----v----------------------v----+
 *        | GatherND (16 pixels)           |
 *        +----------------+---------------+
 *                         |
 *                   +-----v--------+
 *                   | Multiply     |
 *                   | by weights   |
 *                   +-----+--------+
 *                         |
 *                   +-----v--------+
 *                   | ReduceSum    |
 *                   | (weighted)   |
 *                   +-----+--------+
 *                         |
 *                +--------v---------------+
 *                | Output[N,C,H_out,W_out]|
 *                +------------------------+
 *
 * Bicubic interpolation uses a 4x4 pixel neighborhood with cubic weighting
 * functions, providing smoother results than bilinear at higher computational cost.
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

}  // namespace ov::intel_cpu