// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "pass.hpp"

#include "snippets/op/loop.hpp"
#include "snippets/lowered/loop_manager.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

/**
 * @interface InsertTailLoop
 * @brief Injects tail-processing loop after a vector loop if required.
 *  Additional optimizations are performed if a loop body is executed only once.
 * @ingroup snippets
 */
class InsertTailLoop : public Pass {
public:
    OPENVINO_RTTI("InsertTailLoop", "Pass")
    bool run(LinearIR& linear_ir) override;
    static LinearIR::constExprIt insert_copy_loop(LinearIR& linear_ir, const size_t loop_id, const LinearIR::constExprIt& insert_pos);

    static constexpr size_t existing_subtensor_value = SIZE_MAX;
    static void propagate_updated_subtensor_through_loop(const LinearIR& linear_ir,
                                                         const LinearIR::LoopManager::LoopInfoPtr& loop_info,
                                                         LinearIR::container::const_iterator begin,
                                                         LinearIR::container::const_iterator end,
                                                         const size_t new_dim_value = existing_subtensor_value);

private:
    static void create_tail_loop(LinearIR& linear_ir,
                                 LinearIR::constExprIt begin,
                                 LinearIR::constExprIt end,
                                 const std::shared_ptr<op::LoopEnd>& loop_end,
                                 bool need_vector_loop,
                                 size_t tail_size);
    static void tail_transformations(LinearIR& linear_ir,
                                     LinearIR::constExprIt tail_begin,
                                     LinearIR::constExprIt tail_end,
                                     size_t tail_size);
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
