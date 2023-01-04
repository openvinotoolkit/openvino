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
 * @interface InsertTailLoop
 * @brief Injects tail-processing loop after a vector loop if required.
 *  Additional optimizations are performed if a loop body is executed only once.
 * @ingroup snippets
 */
class InsertTailLoop : public LinearIRTransformation {
    static void tail_transformations(LoweredExprIR& linear_ir,
                              LoweredExprIR::container::const_iterator tail_begin,
                              LoweredExprIR::container::const_iterator tail_end,
                              size_t tail_size);
public:
    OPENVINO_RTTI("InsertTailLoop", "LinearIRTransformation")
    bool run(LoweredExprIR& linear_ir) override;
};

} // namespace lowered
} // namespace pass
} // namespace snippets
} // namespace ngraph
