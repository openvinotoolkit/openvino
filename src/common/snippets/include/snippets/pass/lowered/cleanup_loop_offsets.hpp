// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "linear_IR_transformation.hpp"

namespace ngraph {
namespace snippets {
namespace pass {
namespace lowered {

/**
 * @interface CleanupLoopOffsets
 * @brief Loops are inserted with finalization offsets that reset all managed pointers to their initial values.
 *        This transformation "fuses" the offsets with an outer loop's ptr_increments, and zeroes the offsets before Results.
 * @ingroup snippets
 */
class CleanupLoopOffsets : public LinearIRTransformation {
public:
    OPENVINO_RTTI("CleanupLoopOffsets", "LinearIRTransformation")
    bool run(LoweredExprIR& linear_ir) override;
};

} // namespace lowered
} // namespace pass
} // namespace snippets
} // namespace ngraph
