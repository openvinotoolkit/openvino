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
 * @interface LoadStoreInsertion
 * @brief The pass inserts Load and Store expressions in Linear IR after Parameters, Buffers and before Results, Buffers accordingly.
 *        Note: The pass should be called after LoopFusion and BufferInsertion passes to have all possible data expressions.
 * @param m_vector_size - the count of elements for loading/storing
 * @ingroup snippets
 */
class LoadStoreInsertion : public LinearIRTransformation {
public:
    explicit LoadStoreInsertion(size_t vector_size);
    OPENVINO_RTTI("LoadStoreInsertion", "LinearIRTransformation")
    bool run(LoweredExprIR& linear_ir) override;

private:
    bool insert_load(LoweredExprIR& linear_ir, const LoweredExprIR::constExprIt& data_expr_it);
    bool insert_store(LoweredExprIR& linear_ir, const LoweredExprIR::constExprIt& data_expr_it);
    void update_loops(const LoweredExprIR::LoweredLoopManagerPtr& loop_manager, const std::vector<size_t>& loop_ids,
                      const LoweredExprPort& actual_port, const std::vector<LoweredExprPort>& target_ports, bool is_entry = true);
    void update_loop(const LoweredExprIR::LoweredLoopManager::LoweredLoopInfoPtr& loop_info,
                     const LoweredExprPort& actual_port, const std::vector<LoweredExprPort>& target_ports, bool is_entry = true);
    std::vector<size_t> get_loops_for_update(const std::vector<size_t>& loop_ids, size_t loop_id);

    size_t m_vector_size;
};

} //namespace lowered
} // namespace pass
} // namespace snippets
} // namespace ngraph
