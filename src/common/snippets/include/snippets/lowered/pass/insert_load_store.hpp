// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "pass.hpp"

#include "snippets/lowered/loop_manager.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

/**
 * @interface InsertLoadStore
 * @brief The pass inserts Load and Store expressions in Linear IR after Parameters, Buffers and before Results, Buffers accordingly.
 *        Note: The pass should be called after FuseLoops and InsertBuffers passes to have all possible data expressions.
 * @param m_vector_size - the count of elements for loading/storing
 * @ingroup snippets
 */
class InsertLoadStore : public RangedPass {
public:
    OPENVINO_RTTI("InsertLoadStore", "", RangedPass);
    explicit InsertLoadStore(size_t vector_size);
    bool run(LinearIR& linear_ir, lowered::LinearIR::constExprIt begin, lowered::LinearIR::constExprIt end) override;

private:
    size_t get_count(const ExpressionPort& port) const;
    bool insert_load(LinearIR& linear_ir, const LinearIR::constExprIt& data_expr_it);
    bool insert_store(LinearIR& linear_ir, const LinearIR::constExprIt& data_expr_it);

    size_t m_vector_size;
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
