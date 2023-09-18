// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "pass.hpp"
#include "snippets/snippets_isa.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

/**
 * @interface IBufferPass
 * @brief The common interface class for derived classes for work with Buffer ops
 * @ingroup snippets
 */
class IBufferPass : public Pass {
public:
    OPENVINO_RTTI("IBufferPass", "Pass")
    IBufferPass() : Pass() {}

    /**
     * @brief Set offset to Buffer op and propagates its to the connected memory access ops
     * @param buffer_expr expression with Buffer op
     * @param offset offset in common buffer scratchpad
     */
    static void set_buffer_offset(const ExpressionPtr& buffer_expr, const size_t offset);

    using BufferCluster = std::set<ExpressionPtr>;
    using BufferClusters = std::vector<BufferCluster>;
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
