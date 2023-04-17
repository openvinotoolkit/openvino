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
 * @interface CleanupLoopOffsets
 * @brief Loops are inserted with finalization offsets that reset all managed pointers to their initial values.
 *        This transformation "fuses" the offsets with an outer loop's ptr_increments, and zeroes the offsets before Results.
 * @ingroup snippets
 */
class CleanupLoopOffsets : public Transformation {
public:
    OPENVINO_RTTI("CleanupLoopOffsets", "Transformation")
    bool run(LinearIR& linear_ir) override;
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ngraph
