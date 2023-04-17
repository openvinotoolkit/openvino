// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "transformation.hpp"

namespace ngraph {
namespace snippets {
namespace lowered {
namespace pass {

/**
 * @interface ResetBuffers
 * @brief The pass `fuses` (reset) ptr increments and finalization offsets for ports of Loop
 *        with the same Buffers (with the same ID) to avoid double ptr shifts
 *        Note: Buffer always employ inplace logics by default. It means that if a loop has both
 *              an input and an output connected to Buffers, the corresponding register should nevertheless be
 *              incremented only once (because when the input reg is incremented, output incremented automatically).
 *              This condition should be removed when Buffers stop being inplace by default.
 * @ingroup snippets
 */
class ResetBuffers: public Transformation {
public:
    OPENVINO_RTTI("ResetBuffers", "Transformation")
    ResetBuffers() = default;

    bool run(LinearIR& linear_ir) override;

private:
    bool reuse_buffer_increments(const LinearIR& linear_ir, const ExpressionPtr& loop_end_expr);
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ngraph
