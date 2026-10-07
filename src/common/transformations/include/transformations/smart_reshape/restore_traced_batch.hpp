// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API RestoreTracedBatch;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief Restores the batch that tracing froze to one in a `window_reverse` subgraph.
 *
 * A traced `window_reverse` may compute its batch as `int(windows.shape[0] / (H * W / ws / ws))`, which tracing with
 * batch one turns into a constant. Only this subgraph is matched: a leading constant one cannot be told apart from an
 * intentional collapse in general, so the pass is not a generic batch restoration.
 *
 * In both pinned targets the baked constant one is replaced with the batch read from the `Parameter` shape, reusing
 * the `Gather(ShapeOf(Parameter), 0)` that the last `Reshape` already takes it from, optionally through a `Convert`.
 * The targets are edited in place: the baked dimension belongs to the shape tensor, so every `Reshape` reading it
 * needs the same restored batch.
 *
 * ## Before
 *
 *           windows [B * nW, ws, ws, C]
 *                      |
 *     Reshape(Concat(1, H / ws, W / ws, ws, ws, -1))     (batch pinned by tracing)
 *                      |
 *          Transpose(axis 0 preserved)
 *                      |
 *          Reshape(Concat(1, H, W, -1))                  (batch pinned by tracing)
 *                      |
 *          Roll(non-leading axes)                        (optional, shifted windows)
 *                      |
 *     Reshape(Concat(Gather(ShapeOf(Parameter), 0), H * W, C))
 *
 * Every node between the two pinned `Reshape`s and the last one must have a single consumer.
 *
 * ## After
 *
 *           windows [B * nW, ws, ws, C]
 *                      |
 *     Reshape(Concat(Gather(ShapeOf(Parameter), 0), H / ws, W / ws, ws, ws, -1))
 *                      |
 *          Transpose(axis 0 preserved)
 *                      |
 *          Reshape(Concat(Gather(ShapeOf(Parameter), 0), H, W, -1))
 *                      |
 *          Roll(non-leading axes)
 *                      |
 *     Reshape(Concat(Gather(ShapeOf(Parameter), 0), H * W, C))
 */
class ov::pass::RestoreTracedBatch : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("RestoreTracedBatch");
    RestoreTracedBatch();
};
