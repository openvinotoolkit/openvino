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
 * @brief Decomposes Softmax to a range of low-level operations on linear IR
 * @ingroup snippets
 */
class SoftmaxDecomposition : public LinearIRTransformation {
public:
    explicit SoftmaxDecomposition(size_t vector_size);
    OPENVINO_RTTI("SoftmaxDecomposition", "LinearIRTransformation")
    bool run(LoweredExprIR& linear_ir) override;

private:
    size_t m_vector_size;
};

} //namespace lowered
} // namespace pass
} // namespace snippets
} // namespace ngraph
