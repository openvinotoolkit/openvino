// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pattern/multi_matcher.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API RestoreTracedBatch;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief Restores a Reshape target leading dimension that model tracing froze to one.
 *
 * Tracing with batch one turns expressions like `int(x.shape[0])` into constants, so a Reshape keeps a leading one
 * while a `-1` absorbs the batch. When a Reshape taking the batch from the input shape follows such pins, separated
 * only by Transpose and Roll keeping the leading axis, each constant is replaced with that `Gather(ShapeOf(parameter),
 * 0)`.
 *
 * ## Before
 *
 *         Parameter [B, 4]
 *                |
 *               ...          Constant(1)   Constant(-1)
 *                |                |             |
 *                |                +---Concat----+
 *                |                       |
 *             Reshape <------------------+
 *            [1, 4 * B]       (batch pinned by tracing)
 *                |
 *         Transpose / Roll    (optional, keeping axis 0)
 *                |      Gather(ShapeOf(Parameter), 0)   Constant(-1)
 *                |                    |                      |
 *                |                    +--------Concat--------+
 *                |                               |
 *             Reshape <--------------------------+
 *              [B, 4]         (batch rebuilt)
 *
 * ## After
 *
 *         Parameter [B, 4]
 *                |
 *               ...     Gather(ShapeOf(Parameter), 0)   Constant(-1)
 *                |                    |                      |
 *                |                    +--------Concat--------+
 *                |                               |
 *             Reshape <--------------------------+
 *              [B, 4]
 *                |
 *         Transpose / Roll
 *                |
 *             Reshape
 *              [B, 4]
 */
class ov::pass::RestoreTracedBatch : public ov::pass::MultiMatcher {
public:
    OPENVINO_RTTI("RestoreTracedBatch", "0", ov::pass::MultiMatcher);
    RestoreTracedBatch();
};
