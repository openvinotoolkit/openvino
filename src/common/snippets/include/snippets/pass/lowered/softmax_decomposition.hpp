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
 * @interface SoftmaxDecomposition
 * @brief Decomposes snippets::op::Softmax to a range of low-level operations on linear IR
 * @ingroup snippets
 */
class SoftmaxDecomposition : public LinearIRTransformation {
    size_t m_vector_size;
    size_t m_buffer_allocation_rank;
public:
    explicit SoftmaxDecomposition(size_t vector_size, size_t buffer_allocation_rqnk);
    OPENVINO_RTTI("SoftmaxDecomposition", "LinearIRTransformation")
    bool run(LoweredExprIR& linear_ir) override;
};

} //namespace lowered
} // namespace pass
} // namespace snippets
} // namespace ngraph
