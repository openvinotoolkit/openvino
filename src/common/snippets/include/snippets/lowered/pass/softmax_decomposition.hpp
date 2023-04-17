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
 * @interface SoftmaxDecomposition
 * @brief Decomposes Softmax to a range of low-level operations on linear IR
 * @ingroup snippets
 */
class SoftmaxDecomposition : public Transformation {
public:
    explicit SoftmaxDecomposition(size_t vector_size);
    OPENVINO_RTTI("SoftmaxDecomposition", "Transformation")
    bool run(LinearIR& linear_ir) override;

private:
    size_t m_vector_size;
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ngraph
