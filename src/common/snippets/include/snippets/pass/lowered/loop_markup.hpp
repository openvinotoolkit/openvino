// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "linear_IR_transformation.hpp"
#include "snippets/tensor_descriptor.hpp"

namespace ngraph {
namespace snippets {
namespace pass {
namespace lowered {

/**
 * @interface LoopMarkup
 * @brief The pass marks expressions with Loop IDs.
 *        The pass iterates expression by expression till the following conditions:
 *          - the layouts and subtensors them are the same
 *          - the consumer of the expression is explicitly after this expression - the pass marks the branches
 * @ingroup snippets
 */
class LoopMarkup : public LinearIRTransformation {
public:
    OPENVINO_RTTI("LoopMarkup", "LinearIRTransformation")
    LoopMarkup(size_t vector_size);
    bool run(LoweredExprIR& linear_ir) override;

private:
    size_t m_vector_size;
};

} // namespace lowered
} // namespace pass
} // namespace snippets
} // namespace ngraph
