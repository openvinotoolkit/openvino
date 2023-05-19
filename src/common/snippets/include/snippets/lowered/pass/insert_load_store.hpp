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
class InsertLoadStore : public Pass {
public:
    explicit InsertLoadStore(size_t vector_size);
    OPENVINO_RTTI("InsertLoadStore", "Pass")
    bool run(LinearIR& linear_ir) override;

private:
    bool insert_load(LinearIR& linear_ir, const LinearIR::constExprIt& data_expr_it);
    bool insert_store(LinearIR& linear_ir, const LinearIR::constExprIt& data_expr_it);
    void update_loops(const LinearIR::LoopManagerPtr& loop_manager, const std::vector<size_t>& loop_ids,
                      const ExpressionPort& actual_port, const std::vector<ExpressionPort>& target_ports, bool is_entry = true);
    void update_loop(const LinearIR::LoopManager::LoopInfoPtr& loop_info,
                     const ExpressionPort& actual_port, const std::vector<ExpressionPort>& target_ports, bool is_entry = true);
    size_t get_count(const PortDescriptorPtr& port_desc) const;

    size_t m_vector_size;
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
