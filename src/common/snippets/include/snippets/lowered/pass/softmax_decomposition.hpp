// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "pass.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

/**
 * @interface SoftmaxDecomposition
 * @brief Decomposes Softmax to a range of low-level operations on linear IR
 * @ingroup snippets
 */
class SoftmaxDecomposition : public Pass {
public:
    OPENVINO_RTTI("SoftmaxDecomposition", "Pass")
    explicit SoftmaxDecomposition(size_t vector_size);
    bool run(LinearIR& linear_ir) override;

private:
    size_t m_vector_size;
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
