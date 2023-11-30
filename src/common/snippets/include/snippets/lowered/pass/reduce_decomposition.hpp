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
 * @interface ReduceDecomposition
 * @brief Decomposes supported Reduce operations to a range of low-level operations on linear IR
 * @ingroup snippets
 */
class ReduceDecomposition : public Pass {
public:
    OPENVINO_RTTI("ReduceDecomposition", "Pass")
    explicit ReduceDecomposition(size_t vector_size);
    bool run(LinearIR& linear_ir) override;

private:
    size_t m_vector_size;
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
