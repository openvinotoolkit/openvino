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
 * @interface InsertTailLoop
 * @brief Injects tail-processing loop after a vector loop if required.
 *  Additional optimizations are performed if a loop body is executed only once.
 * @ingroup snippets
 */
class InsertTailLoop : public Transformation {
    static void tail_transformations(LinearIR& linear_ir,
                                     LinearIR::container::const_iterator tail_begin,
                                     LinearIR::container::const_iterator tail_end,
                                     size_t tail_size);
public:
    OPENVINO_RTTI("InsertTailLoop", "Transformation")
    bool run(LinearIR& linear_ir) override;
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ngraph
